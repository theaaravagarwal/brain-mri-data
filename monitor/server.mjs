import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { chmod, mkdir, readFile, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
const runtime = join(here, ".runtime");
const controlRuntime = `/tmp/brain-mri-monitor-${typeof process.getuid === "function" ? process.getuid() : "user"}`;
const pollIntervalMs = Number(process.env.MONITOR_POLL_MS || 10_000);
const requestTimeoutMs = Number(process.env.MONITOR_SSH_TIMEOUT_MS || 8_000);
const maxOutputBytes = 512 * 1024;

export const HOSTS = Object.freeze({
  nvidia: {
    target: validateTarget(process.env.MONITOR_NVIDIA_TARGET || "theaa@100.64.0.3"),
    collector: join(here, "collectors", "nvidia.py")
  },
  amd: {
    target: validateTarget(process.env.MONITOR_AMD_TARGET || "b@100.64.0.5"),
    collector: join(here, "collectors", "amd.py")
  }
});

export function validateTarget(target) {
  if (!/^[A-Za-z0-9._-]+@[A-Za-z0-9.:[\]-]+$/.test(target) || target.startsWith("-")) {
    throw new Error(`Invalid fixed SSH target: ${target}`);
  }
  return target;
}

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
    return [id, {
      reachable: host.reachable,
      stale: !host.reachable || age > pollIntervalMs * 2.5,
      lastSuccessAt: host.lastSuccessAt,
      staleSince: host.staleSince,
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
    "--", config.target, "python3", "-"
  ];
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
        const data = JSON.parse(stdout.toString("utf8"));
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
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
  });
  createReadStream(target).pipe(res);
}

export function requestHandler(req, res) {
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
  } else {
    schedule("nvidia");
    schedule("amd");
  }
  const port = Number(process.env.MONITOR_PORT || 4173);
  createServer(requestHandler).listen(port, "127.0.0.1", () => {
    console.log(`Brain MRI monitor: http://127.0.0.1:${port}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
