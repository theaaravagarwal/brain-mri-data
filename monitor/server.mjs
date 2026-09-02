import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { chmod, mkdir, readFile, stat } from "node:fs/promises";
import { createReadStream, readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isStudyApiPath, studyService } from "./study-service.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
const runtime = join(here, ".runtime");
const resources = join(here, "resources");
const controlRuntime = `/tmp/brain-mri-monitor-${typeof process.getuid === "function" ? process.getuid() : "user"}`;
const pollIntervalMs = Number(process.env.MONITOR_POLL_MS || 10_000);
const requestTimeoutMs = Number(process.env.MONITOR_SSH_TIMEOUT_MS || 8_000);
const maxOutputBytes = 512 * 1024;
const resourceSchemas = Object.freeze({ evidence: "research-evidence/v1", explanation: "research-explanation/v1", proposals: "research-proposals/v1" });
const sha256 = /^[0-9a-f]{64}$/;

function finiteTree(value) {
  if (typeof value === "number" && !Number.isFinite(value)) throw new Error("non-finite number");
  if (Array.isArray(value)) value.forEach(finiteTree);
  else if (value && typeof value === "object") Object.values(value).forEach(finiteTree);
}

export function parseStrictJson(text) {
  let position = 0;
  const whitespace = () => { while (/\s/.test(text[position] || "")) position += 1; };
  const string = () => {
    const start = position++;
    while (position < text.length) {
      if (text[position] === "\\") { position += 2; continue; }
      if (text[position++] === '"') return JSON.parse(text.slice(start, position));
    }
    throw new Error("unterminated JSON string");
  };
  const value = () => {
    whitespace();
    if (text[position] === "{") {
      position += 1;
      const keys = new Set();
      whitespace();
      if (text[position] === "}") { position += 1; return; }
      while (position < text.length) {
        whitespace();
        if (text[position] !== '"') throw new Error("invalid JSON object key");
        const key = string();
        if (keys.has(key)) throw new Error(`duplicate JSON key: ${key}`);
        keys.add(key);
        whitespace();
        if (text[position++] !== ":") throw new Error("invalid JSON object separator");
        value();
        whitespace();
        const separator = text[position++];
        if (separator === "}") return;
        if (separator !== ",") throw new Error("invalid JSON object");
      }
      throw new Error("unterminated JSON object");
    }
    if (text[position] === "[") {
      position += 1;
      whitespace();
      if (text[position] === "]") { position += 1; return; }
      while (position < text.length) {
        value();
        whitespace();
        const separator = text[position++];
        if (separator === "]") return;
        if (separator !== ",") throw new Error("invalid JSON array");
      }
      throw new Error("unterminated JSON array");
    }
    if (text[position] === '"') { string(); return; }
    const start = position;
    while (position < text.length && !/[\s,\]}]/.test(text[position])) position += 1;
    if (start === position) throw new Error("invalid JSON value");
    JSON.parse(text.slice(start, position));
  };
  value();
  whitespace();
  if (position !== text.length) throw new Error("trailing JSON content");
  return JSON.parse(text);
}

export function validateResource(kind, value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || value.schemaVersion !== resourceSchemas[kind]) throw new Error("unsupported resource schema");
  if (typeof value.generatedAt !== "string" || Number.isNaN(Date.parse(value.generatedAt))) throw new Error("invalid generatedAt");
  if (!value.source || !["fresh", "stale", "unavailable", "rejected"].includes(value.source.status)) throw new Error("invalid source status");
  if (value.artifactDigest !== null && (typeof value.artifactDigest !== "string" || !sha256.test(value.artifactDigest))) throw new Error("invalid artifact digest");
  finiteTree(value);
  if (value.data === null) return value;
  if (kind === "evidence") {
    const status = value.data.promotion?.status;
    if (value.data.studyId !== "glioma" || value.data.automaticPromotion !== false || !["blocked", "pending", "rejected", "selected"].includes(status) || !Array.isArray(value.data.promotion.missing)) throw new Error("invalid evidence decision contract");
    if (!value.data.baseline || !Array.isArray(value.data.candidates)) throw new Error("invalid evidence arms");
    if (status === "rejected") {
      if (value.data.reviewStatus !== "candidate_rejected_baseline_retained" || value.data.selectedModel?.variantId !== "baseline" || value.data.selectedModel?.readiness !== "internal_research_only") throw new Error("rejected evidence must retain the baseline");
      if (!Array.isArray(value.data.baseline.seedMetrics) || value.data.baseline.seedMetrics.length !== 3 || !Array.isArray(value.data.candidates[0]?.seedMetrics) || value.data.candidates[0].seedMetrics.length !== 3) throw new Error("complete evidence requires three seeds per arm");
      if (!Array.isArray(value.data.comparison?.effects) || value.data.comparison.effects.length < 1) throw new Error("complete evidence requires paired effects");
    }
  }
  if (kind === "explanation" && (!value.data.deterministic || !value.data.llm)) throw new Error("invalid explanation resource");
  if (kind === "proposals" && (!Array.isArray(value.data.jobs) || !Array.isArray(value.data.proposals))) throw new Error("invalid proposal resource");
  for (const proposal of value.data?.proposals || []) if (proposal.executed !== false) throw new Error("proposal execution is forbidden");
  return value;
}

function resourcePayload(kind) {
  const scenario = process.env.MONITOR_FIXTURE_SCENARIO;
  const path = scenario ? join(here, "fixtures", `${kind}-${scenario}.json`) : join(resources, `${kind}.json`);
  try { return validateResource(kind, parseStrictJson(readFileSync(path, "utf8"))); }
  catch (error) {
    if (scenario) return { schemaVersion: resourceSchemas[kind], generatedAt: new Date().toISOString(), source: { status: "rejected", staleSince: null, error: `Fixture rejected: ${error.message}` }, artifactDigest: null, data: null };
    throw error;
  }
}

function sendJson(req, res, payload) {
  const body = JSON.stringify(payload), etag = etagFor(body);
  if (req.headers["if-none-match"] === etag) return res.writeHead(304, { ETag: etag, "Cache-Control": "no-store" }).end();
  res.writeHead(200, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body), "Cache-Control": "no-store", ETag: etag, "X-Content-Type-Options": "nosniff" }).end(body);
}

export const HOSTS = Object.freeze({
  nvidia: {
    target: validateTarget(process.env.MONITOR_NVIDIA_TARGET || "software@100.64.0.1"),
    collector: join(here, "collectors", "nvidia.py")
  },
  amd: {
    target: validateTarget(process.env.MONITOR_AMD_TARGET || "theaa@100.64.0.3"),
    collector: join(here, "collectors", "amd.py")
  }
});

export function validateTarget(target) {
  if (!/^[A-Za-z0-9._-]+@[A-Za-z0-9.:[\]-]+$/.test(target) || target.startsWith("-")) {
    throw new Error(`Invalid fixed SSH target: ${target}`);
  }
  return target;
}

export function validateIdentity(identity) {
  if (identity === undefined) return null;
  if (!identity.startsWith("/") || identity.includes("\0") || identity.includes("\n")) {
    throw new Error("MONITOR_SSH_IDENTITY must be an absolute path");
  }
  return identity;
}

export function parseCollectorOutput(output) {
  const lines = String(output).split("\n").map(line => line.trim()).filter(Boolean);
  if (lines.length === 0) throw new Error("empty collector output");
  return JSON.parse(lines.at(-1));
}

const sshIdentity = validateIdentity(process.env.MONITOR_SSH_IDENTITY);

function emptyHostState() {
  return {
    data: null,
    lastSuccessAt: null,
    staleSince: null,
    latencyMs: null,
    error: null,
    reachable: false,
    failures: 0,
    nextAttemptAt: new Date().toISOString()
  };
}

const state = {
  updatedAt: new Date().toISOString(),
  hosts: { nvidia: emptyHostState(), amd: emptyHostState() }
};

export function publicSnapshot(source = state) {
  const now = Date.now();
  const hosts = Object.fromEntries(Object.entries(source.hosts).map(([id, host]) => {
    const age = host.lastSuccessAt ? now - Date.parse(host.lastSuccessAt) : Infinity;
    const stale = !host.reachable || age > pollIntervalMs * 2.5;
    return [id, {
      reachable: host.reachable,
      stale,
      lastSuccessAt: host.lastSuccessAt,
      staleSince: stale ? host.staleSince || (host.lastSuccessAt ? new Date(Date.parse(host.lastSuccessAt) + pollIntervalMs * 2.5).toISOString() : null) : null,
      latencyMs: host.latencyMs,
      nextAttemptAt: host.nextAttemptAt,
      error: host.error,
      data: host.data
    }];
  }));
  return {
    schemaVersion: 1,
    generatedAt: source.updatedAt,
    pollIntervalMs,
    hosts
  };
}

export function etagFor(body) {
  return `"${createHash("sha256").update(body).digest("base64url")}"`;
}

export async function runCollector(id, config, options = {}) {
  const script = await readFile(config.collector);
  const spawnImpl = options.spawnImpl || spawn;
  const timeoutMs = options.timeoutMs || requestTimeoutMs;
  const args = [
    "-T",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=5",
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=1",
    "-o", "Compression=yes",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=600",
    "-o", `ControlPath=${join(controlRuntime, "ssh-%C")}`,
  ];
  if (sshIdentity) args.push("-i", sshIdentity);
  args.push("--", config.target, "python3", "-");
  const started = performance.now();
  return await new Promise((resolvePromise, reject) => {
    const child = spawnImpl("ssh", args, { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = Buffer.alloc(0);
    let stderr = Buffer.alloc(0);
    let overflow = false;
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`${id} collection timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    child.stdout.on("data", chunk => {
      if (stdout.length + chunk.length > maxOutputBytes) {
        overflow = true;
        child.kill("SIGTERM");
        return;
      }
      stdout = Buffer.concat([stdout, chunk]);
    });
    child.stderr.on("data", chunk => {
      if (stderr.length < 16_384) stderr = Buffer.concat([stderr, chunk]).subarray(0, 16_384);
    });
    child.on("error", error => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", code => {
      clearTimeout(timer);
      if (overflow) return reject(new Error(`${id} collector exceeded ${maxOutputBytes} bytes`));
      if (code !== 0) return reject(new Error(`${id} SSH failed (${code}): ${stderr.toString().trim() || "no detail"}`));
      try {
        const data = parseCollectorOutput(stdout.toString("utf8"));
        if (data?.id !== id || typeof data?.label !== "string" || !data?.disk) throw new Error("unexpected collector schema");
        resolvePromise({ data, latencyMs: Math.round(performance.now() - started) });
      } catch (error) {
        reject(new Error(`${id} returned invalid JSON: ${error.message}`));
      }
    });
    child.stdin.end(script);
  });
}

function safeError(error) {
  return String(error?.message || error).replaceAll(here, "<monitor>").slice(0, 500);
}

function schedule(id, delay = 0) {
  const host = state.hosts[id];
  host.nextAttemptAt = new Date(Date.now() + delay).toISOString();
  setTimeout(async () => {
    try {
      const result = await runCollector(id, HOSTS[id]);
      const now = new Date().toISOString();
      Object.assign(host, {
        data: result.data,
        lastSuccessAt: now,
        staleSince: null,
        latencyMs: result.latencyMs,
        error: null,
        reachable: true,
        failures: 0
      });
      state.updatedAt = now;
      schedule(id, pollIntervalMs);
    } catch (error) {
      const now = new Date().toISOString();
      host.staleSince ||= now;
      host.error = safeError(error);
      host.reachable = false;
      host.failures += 1;
      state.updatedAt = now;
      schedule(id, Math.min(pollIntervalMs * (2 ** Math.min(host.failures - 1, 3)), 60_000));
    }
  }, delay).unref();
}

const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".map": "application/json; charset=utf-8"
};

export function resolveStaticPath(urlPath, root = dist) {
  const decoded = decodeURIComponent(urlPath.split("?", 1)[0]);
  const relative = normalize(decoded).replace(/^([/\\])+/, "");
  const candidate = resolve(root, relative || "index.html");
  if (candidate !== resolve(root) && !candidate.startsWith(resolve(root) + "/")) return null;
  return candidate;
}

async function serveStatic(req, res) {
  let target = resolveStaticPath(req.url || "/");
  if (!target) {
    res.writeHead(400).end("Bad path");
    return;
  }
  try {
    const info = await stat(target);
    if (info.isDirectory()) target = join(target, "index.html");
    await stat(target);
  } catch {
    target = join(dist, "index.html");
  }
  res.writeHead(200, {
    "Content-Type": contentTypes[extname(target)] || "application/octet-stream",
    "Cache-Control": target.endsWith("index.html") ? "no-cache" : "public, max-age=31536000, immutable",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
  });
  createReadStream(target).pipe(res);
}

export function requestHandler(req, res) {
  if (isStudyApiPath(req.url)) {
    void studyService.handle(req, res);
    return;
  }
  if (req.method === "GET" && req.url?.split("?", 1)[0] === "/api/status") {
    const body = JSON.stringify(publicSnapshot());
    const etag = etagFor(body);
    if (req.headers["if-none-match"] === etag) {
      res.writeHead(304, { ETag: etag, "Cache-Control": "no-store" }).end();
      return;
    }
    res.writeHead(200, {
      "Content-Type": "application/json; charset=utf-8",
      "Content-Length": Buffer.byteLength(body),
      "Cache-Control": "no-store",
      ETag: etag,
      "X-Content-Type-Options": "nosniff"
    }).end(body);
    return;
  }
  if (req.method === "GET" && ["/api/evidence", "/api/explanation", "/api/proposals"].includes(req.url?.split("?", 1)[0])) {
    const kind = req.url.split("/", 3)[2];
    try { sendJson(req, res, resourcePayload(kind)); }
    catch (error) { res.writeHead(503, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" }).end(JSON.stringify({ error: "contract_error", detail: String(error.message).slice(0, 240) })); }
    return;
  }
  if (req.url?.startsWith("/api/")) {
    res.writeHead(404, { "Content-Type": "application/json" }).end('{"error":"not found"}');
    return;
  }
  void serveStatic(req, res);
}

async function main() {
  await mkdir(runtime, { recursive: true, mode: 0o700 });
  await chmod(runtime, 0o700);
  await mkdir(controlRuntime, { recursive: true, mode: 0o700 });
  await chmod(controlRuntime, 0o700);
  await studyService.initialize();
  const controlInfo = await stat(controlRuntime);
  if (!controlInfo.isDirectory() || (typeof process.getuid === "function" && controlInfo.uid !== process.getuid())) {
    throw new Error(`Unsafe SSH control directory: ${controlRuntime}`);
  }
  if (process.env.MONITOR_FIXTURE === "1") {
    const fixture = JSON.parse(await readFile(join(here, "fixtures", "snapshot.json"), "utf8"));
    state.hosts = fixture.hosts;
    const now = new Date().toISOString();
    for (const host of Object.values(state.hosts)) {
      host.lastSuccessAt = now;
      host.nextAttemptAt = now;
    }
    state.updatedAt = now;
    setInterval(() => {
      const refreshedAt = new Date().toISOString();
      state.updatedAt = refreshedAt;
      for (const host of Object.values(state.hosts)) {
        host.lastSuccessAt = refreshedAt;
        host.nextAttemptAt = new Date(Date.now() + pollIntervalMs).toISOString();
      }
    }, pollIntervalMs).unref();
  } else {
    schedule("nvidia");
    schedule("amd");
  }
  const port = Number(process.env.MONITOR_PORT || 4173);
  const bindHost = process.env.MONITOR_BIND_HOST || "127.0.0.1";
  if (bindHost !== "127.0.0.1" && !/^100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?:\d{1,3})\.(?:\d{1,3})$/.test(bindHost)) {
    throw new Error("MONITOR_BIND_HOST must be 127.0.0.1 or a Tailscale IPv4 address");
  }
  createServer(requestHandler).listen(port, bindHost, () => {
    console.log(`Brain MRI monitor: http://${bindHost}:${port}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
