import Busboy from "busboy";
import { createHash, randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, createWriteStream } from "node:fs";
import { access, chmod, copyFile, mkdir, open, readFile, readdir, rm, stat } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { finished } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const defaultRepoRoot = resolve(here, "..");
const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const EXPECTED_CHECKPOINT_SHA256 = "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5";
const MODALITIES = Object.freeze({ t1: "0000", t1ce: "0001", t2: "0002", flair: "0003" });
const DEMO_MODALITY_SUFFIXES = Object.freeze({
  t1: ["_t1", "-t1n"],
  t1ce: ["_t1ce", "-t1c"],
  t2: ["_t2", "-t2w"],
  flair: ["_flair", "-t2f"]
});
const EVALUATION_DEMO_SCOPES = new Set(["development_validation", "external_public"]);
const MAX_FILE_BYTES = 512 * 1024 * 1024;
const MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024;
const RETENTION_MS = 24 * 60 * 60 * 1000;
const MAX_PROCESS_OUTPUT = 256 * 1024;
const ARTIFACTS = Object.freeze({
  segmentation: { file: "research_segmentation.nii.gz", type: "application/gzip" },
  receipt: { file: "receipt.json", type: "application/json; charset=utf-8" },
  explanation: { file: "explanation.json", type: "application/json; charset=utf-8" }
});

function publicJob(job) {
  return {
    schemaVersion: "research-study-job/v1",
    jobId: job.jobId,
    state: job.state,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
    expiresAt: job.expiresAt,
    validation: job.validation ?? null,
    result: job.result ?? null,
    explanation: job.explanation ?? null,
    error: job.error ?? null,
    artifacts: job.state === "succeeded" ? Object.keys(ARTIFACTS) : [],
    evaluationSampleScope: job.evaluationSampleScope ?? null
  };
}

function sendJson(res, status, value) {
  const body = JSON.stringify(value);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff"
  }).end(body);
}

function safeProcessError(stderr) {
  const lines = String(stderr).trim().split("\n").filter(Boolean);
  const last = lines.at(-1) || "Study processing failed";
  return last.replace(/^\w+(?:Error|Exception):\s*/, "").replaceAll(defaultRepoRoot, "<research-workspace>").slice(0, 280);
}

function isLoopback(address) {
  return !address || address === "127.0.0.1" || address === "::1" || address === "::ffff:127.0.0.1";
}

function isTailscaleIpv4(address = "") {
  const normalized = address.replace(/^::ffff:/, "");
  const parts = normalized.split(".").map(Number);
  return parts.length === 4 && parts.every(part => Number.isInteger(part) && part >= 0 && part <= 255)
    && parts[0] === 100 && parts[1] >= 64 && parts[1] <= 127;
}

function configuredPublicHosts(bindHost) {
  const configured = (process.env.MONITOR_PUBLIC_HOSTS || bindHost)
    .split(",")
    .map(value => value.trim())
    .filter(Boolean);
  const hosts = new Set(["127.0.0.1", "localhost", bindHost]);
  for (const host of configured) {
    if (host !== "localhost" && host !== "127.0.0.1" && !isTailscaleIpv4(host)) {
      throw new Error("MONITOR_PUBLIC_HOSTS must contain only loopback or Tailscale IPv4 addresses");
    }
    hosts.add(host);
  }
  return hosts;
}

function validateMutationRequest(req, bindHost, publicHosts) {
  const address = req.socket?.remoteAddress || "";
  const tailnetAccess = isTailscaleIpv4(bindHost) && isTailscaleIpv4(address);
  if (!isLoopback(address) && !tailnetAccess) throw Object.assign(new Error("Local or Tailscale access required"), { status: 403 });
  const host = req.headers.host || "";
  let hostname;
  try { hostname = new URL(`http://${host}`).hostname; } catch { throw Object.assign(new Error("Invalid host"), { status: 403 }); }
  if (!publicHosts.has(hostname)) throw Object.assign(new Error("Invalid local or Tailscale host"), { status: 403 });
  const origin = req.headers.origin;
  if (origin) {
    let originHost;
    try { originHost = new URL(origin).host; } catch { throw Object.assign(new Error("Invalid origin"), { status: 403 }); }
    if (originHost !== host) throw Object.assign(new Error("Cross-origin request rejected"), { status: 403 });
  }
  if (req.headers["sec-fetch-site"] && req.headers["sec-fetch-site"] !== "same-origin") {
    throw Object.assign(new Error("Cross-site request rejected"), { status: 403 });
  }
}

async function fileSha256(path) {
  const digest = createHash("sha256");
  const stream = createReadStream(path);
  for await (const chunk of stream) digest.update(chunk);
  return digest.digest("hex");
}

async function runJsonProcess(command, args, { cwd, timeoutMs, spawnImpl = spawn } = {}) {
  return await new Promise((resolvePromise, reject) => {
    const child = spawnImpl(command, args, { cwd, env: process.env, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = Buffer.alloc(0);
    let stderr = Buffer.alloc(0);
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      fn(value);
    };
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref();
      finish(reject, Object.assign(new Error("Study process timed out"), { stderr: stderr.toString() }));
    }, timeoutMs);
    child.stdout.on("data", chunk => {
      if (stdout.length + chunk.length > MAX_PROCESS_OUTPUT) {
        child.kill("SIGTERM");
        finish(reject, new Error("Study process output exceeded its limit"));
      } else stdout = Buffer.concat([stdout, chunk]);
    });
    child.stderr.on("data", chunk => {
      stderr = Buffer.concat([stderr, chunk]).subarray(-MAX_PROCESS_OUTPUT);
    });
    child.on("error", error => finish(reject, error));
    child.on("close", code => {
      if (settled) return;
      if (code !== 0) return finish(reject, Object.assign(new Error(`Study process exited ${code}`), { stderr: stderr.toString() }));
      try { finish(resolvePromise, JSON.parse(stdout.toString("utf8"))); }
      catch { finish(reject, Object.assign(new Error("Study process returned invalid JSON"), { stderr: stderr.toString() })); }
    });
  });
}

export function createStudyService(options = {}) {
  const repoRoot = options.repoRoot || defaultRepoRoot;
  const runtimeRoot = options.runtimeRoot || join(here, ".runtime", "studies");
  const python = options.python || join(repoRoot, ".venv", "bin", "python");
  const runner = options.runner || join(repoRoot, "scripts", "run_4060_research_inference.py");
  const checkpoint = options.checkpoint || join(repoRoot, "runs", "glioma-pilot--cuda-4060--brats--20260828--e100", "best.pt");
  const expectedCheckpointSha256 = options.expectedCheckpointSha256 || EXPECTED_CHECKPOINT_SHA256;
  const demoDirectory = options.demoDirectory || process.env.BRAIN_MRI_DEMO_DIR || null;
  const evaluationDemoDirectory = options.evaluationDemoDirectory || process.env.BRAIN_MRI_EVALUATION_DEMO_DIR || null;
  const evaluationDemoScope = options.evaluationDemoScope || process.env.BRAIN_MRI_EVALUATION_DEMO_SCOPE || "development_validation";
  const benchmarkDirectory = options.benchmarkDirectory || process.env.BRAIN_MRI_EXTERNAL_BENCHMARK_DIR || join(repoRoot, "artifacts", "fixed-segresnet-external");
  if (!EVALUATION_DEMO_SCOPES.has(evaluationDemoScope)) throw new Error("Invalid evaluation demo scope");
  const deviceProbe = options.deviceProbe || (async () => {
    const result = await runJsonProcess(python, ["-c", "import json, torch; print(json.dumps({'cuda': torch.cuda.is_available()}))"], {
      cwd: repoRoot, timeoutMs: 15_000, spawnImpl: options.spawnImpl
    });
    return result.cuda === true;
  });
  const bindHost = options.bindHost || process.env.MONITOR_BIND_HOST || "127.0.0.1";
  const publicHosts = options.publicHosts ? new Set(options.publicHosts) : configuredPublicHosts(bindHost);
  const jobs = new Map();
  const mutationTimes = [];
  let activeJobId = null;
  let initialized = false;
  let capabilities = null;

  async function readExternalBenchmark() {
    for (const filename of ["summary.public.json", "status.public.json"]) {
      try {
        const value = JSON.parse(await readFile(join(benchmarkDirectory, filename), "utf8"));
        if (!["fixed-segresnet-external-summary/v1", "fixed-segresnet-external-status/v1"].includes(value.schema_version)) continue;
        if (!['running', 'complete'].includes(value.status)) continue;
        const encoded = JSON.stringify(value);
        if (/case_token|case_id|native_path|\/home\//i.test(encoded)) throw new Error("Benchmark public artifact contains private fields");
        return value;
      } catch (error) {
        if (error?.code !== "ENOENT") console.error("External benchmark artifact rejected:", error.message);
      }
    }
    return null;
  }

  async function refreshCapabilities() {
    let checkpointStatus = "unavailable";
    let observedCheckpointSha256 = null;
    try {
      await access(python);
      await access(runner);
      observedCheckpointSha256 = await fileSha256(checkpoint);
      checkpointStatus = observedCheckpointSha256 === expectedCheckpointSha256 ? "ready" : "digest_mismatch";
      if (checkpointStatus === "ready" && !await deviceProbe()) checkpointStatus = "unavailable";
    } catch { checkpointStatus = "unavailable"; }
    capabilities = {
      schemaVersion: "research-study-capabilities/v1",
      generatedAt: new Date().toISOString(),
      inference: {
        status: checkpointStatus,
        modelId: "glioma-segresnet-20260828",
        modelScope: "internal_research_only",
        checkpointSha256: expectedCheckpointSha256,
        observedCheckpointSha256,
        outputKind: "binary_whole_lesion_research_segmentation",
        device: process.env.BRAIN_MRI_INFERENCE_DEVICE || "NVIDIA GeForce RTX 4060"
      },
      explanation: {
        deterministic: "available",
        llm: process.env.BRAIN_MRI_LLM_MODEL ? "configured" : "not_configured",
        model: process.env.BRAIN_MRI_LLM_MODEL || null
      },
      demoAvailable: Boolean(demoDirectory),
      evaluationDemoAvailable: Boolean(evaluationDemoDirectory),
      limits: { files: 5, perFileBytes: MAX_FILE_BYTES, totalBytes: MAX_TOTAL_BYTES, retentionHours: 24 }
    };
    capabilities.externalBenchmark = await readExternalBenchmark();
  }

  function getJob(jobId) {
    const job = jobs.get(jobId);
    if (!job) return null;
    if (Date.parse(job.expiresAt) <= Date.now() && job.state !== "running") {
      jobs.delete(jobId);
      void rm(job.directory, { recursive: true, force: true });
      return null;
    }
    return job;
  }

  async function persist(job) {
    const path = join(runtimeRoot, job.jobId, "job.json");
    const handle = await open(path, "w", 0o600);
    try {
      await handle.writeFile(JSON.stringify(publicJob(job), null, 2) + "\n");
      await handle.sync();
    } finally { await handle.close(); }
  }

  async function initialize() {
    if (initialized) return;
    await mkdir(runtimeRoot, { recursive: true, mode: 0o700 });
    await chmod(runtimeRoot, 0o700);
    const now = Date.now();
    for (const entry of await readdir(runtimeRoot, { withFileTypes: true })) {
      if (!entry.isDirectory() || !UUID_V4.test(entry.name)) continue;
      const directory = join(runtimeRoot, entry.name);
      try {
        const saved = JSON.parse(await readFile(join(directory, "job.json"), "utf8"));
        if (Date.parse(saved.expiresAt) <= now || saved.state !== "succeeded") {
          await rm(directory, { recursive: true, force: true });
          continue;
        }
        jobs.set(entry.name, { ...saved, jobId: entry.name, directory });
      } catch { await rm(directory, { recursive: true, force: true }); }
    }
    await refreshCapabilities();
    setInterval(() => {
      for (const jobId of jobs.keys()) getJob(jobId);
    }, 60 * 60 * 1000).unref();
    initialized = true;
  }

  function rateLimit() {
    const cutoff = Date.now() - 15 * 60 * 1000;
    while (mutationTimes.length && mutationTimes[0] < cutoff) mutationTimes.shift();
    if (mutationTimes.length >= 20) throw Object.assign(new Error("Too many study requests"), { status: 429 });
    mutationTimes.push(Date.now());
  }

  async function parseUpload(req, inputDirectory) {
    const received = new Map();
    const writes = [];
    let failure = null;
    let totalBytes = 0;
    const parser = Busboy({
      headers: req.headers,
      limits: { files: 5, fields: 0, parts: 5, fileSize: MAX_FILE_BYTES }
    });
    parser.on("file", (field, stream, info) => {
      const suffix = MODALITIES[field];
      const isReference = field === "reference";
      const lowerName = typeof info.filename === "string" ? info.filename.toLowerCase() : "";
      const extension = lowerName.endsWith(".nii.gz") ? ".nii.gz" : lowerName.endsWith(".nii") ? ".nii" : null;
      const validMime = ["application/octet-stream", "application/gzip", "application/x-gzip", "application/x-nifti", ""].includes(info.mimeType || "");
      if ((!suffix && !isReference) || received.has(field) || !extension || !validMime) {
        failure ||= "Choose one NIfTI file for each scan type and, optionally, one reference outline.";
        stream.resume();
        return;
      }
      const destination = join(inputDirectory, isReference ? `research_reference${extension}` : `research_input_${suffix}${extension}`);
      const output = createWriteStream(destination, { flags: "wx", mode: 0o600 });
      const digest = createHash("sha256");
      let bytes = 0;
      stream.on("data", chunk => {
        bytes += chunk.length;
        totalBytes += chunk.length;
        digest.update(chunk);
        if (totalBytes > MAX_TOTAL_BYTES) failure ||= "The selected study exceeds the 2 GiB upload limit.";
      });
      stream.on("limit", () => { failure ||= `${field.toUpperCase()} exceeds the 512 MiB file limit.`; });
      stream.on("error", error => { failure ||= `Upload interrupted: ${error.message}`; });
      stream.pipe(output);
      writes.push(finished(output).then(() => {
        received.set(field, { bytes, sha256: digest.digest("hex") });
      }));
    });
    parser.on("field", () => { failure ||= "Text fields are not accepted in study uploads."; });
    parser.on("filesLimit", () => { failure ||= "A maximum of five files is allowed."; });
    req.pipe(parser);
    await finished(parser);
    await Promise.all(writes);
    if (failure) throw Object.assign(new Error(failure), { status: 400 });
    const expectedCount = received.has("reference") ? 5 : 4;
    if (received.size !== expectedCount || Object.keys(MODALITIES).some(modality => !received.has(modality))) {
      throw Object.assign(new Error("Select exactly one T1, T1ce, T2, and FLAIR volume."), { status: 400 });
    }
    return Object.fromEntries(received);
  }

  async function upload(req, res) {
    validateMutationRequest(req, bindHost, publicHosts);
    rateLimit();
    await refreshCapabilities();
    if (capabilities.inference.status !== "ready") {
      return sendJson(res, 503, {
        error: "model_unavailable",
        detail: "The exact fixed model is unavailable; the study was not uploaded."
      });
    }
    const contentType = req.headers["content-type"] || "";
    if (!contentType.startsWith("multipart/form-data;")) throw Object.assign(new Error("A multipart study upload is required"), { status: 415 });
    const jobId = randomUUID();
    const directory = join(runtimeRoot, jobId);
    const inputDirectory = join(directory, "input");
    await mkdir(inputDirectory, { recursive: true, mode: 0o700 });
    try {
      const uploadMetadata = await parseUpload(req, inputDirectory);
      const validation = await runJsonProcess(python, [runner, inputDirectory, "--validate-only"], {
        cwd: repoRoot, timeoutMs: 90_000, spawnImpl: options.spawnImpl
      });
      if (validation.schema_version !== "research-study-validation/v1" || validation.status !== "pass") {
        throw new Error("The validation process returned an unsupported contract");
      }
      const now = new Date();
      const job = {
        jobId,
        state: "validated",
        createdAt: now.toISOString(),
        updatedAt: now.toISOString(),
        expiresAt: new Date(now.getTime() + RETENTION_MS).toISOString(),
        directory,
        validation,
        uploadMetadata,
        result: null,
        explanation: null,
        error: null
      };
      jobs.set(jobId, job);
      await persist(job);
      sendJson(res, 201, publicJob(job));
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw Object.assign(new Error(safeProcessError(error.stderr || error.message)), {
        status: error.status || 400
      });
    }
  }

  async function loadDemo(req, res, { evaluation = false } = {}) {
    validateMutationRequest(req, bindHost, publicHosts);
    rateLimit();
    await refreshCapabilities();
    if (capabilities.inference.status !== "ready") {
      return sendJson(res, 503, { error: "model_unavailable", detail: "The model is not ready." });
    }
    const sourceDirectory = evaluation ? evaluationDemoDirectory : demoDirectory;
    if (!sourceDirectory) return sendJson(res, 404, { error: "demo_unavailable" });
    const entries = await readdir(sourceDirectory, { withFileTypes: true });
    const sources = Object.keys(MODALITIES).map(modality => {
      const suffixes = DEMO_MODALITY_SUFFIXES[modality];
      const matches = entries.filter(entry => entry.isFile() && suffixes.some(suffix => entry.name.toLowerCase().endsWith(`${suffix}.nii`) || entry.name.toLowerCase().endsWith(`${suffix}.nii.gz`)));
      if (matches.length !== 1) throw new Error(`Demo ${modality} volume is unavailable`);
      return [modality, join(sourceDirectory, matches[0].name)];
    });
    const reference = evaluation
      ? entries.find(entry => entry.isFile() && /(?:_|-)seg\.nii(?:\.gz)?$/i.test(entry.name))
      : null;
    if (evaluation && !reference) throw new Error("Accuracy sample reference outline is unavailable");
    const jobId = randomUUID();
    const directory = join(runtimeRoot, jobId);
    const inputDirectory = join(directory, "input");
    await mkdir(inputDirectory, { recursive: true, mode: 0o700 });
    try {
      for (const [modality, source] of sources) {
        const extension = source.toLowerCase().endsWith(".nii.gz") ? ".nii.gz" : ".nii";
        const destination = join(inputDirectory, `research_input_${MODALITIES[modality]}${extension}`);
        await copyFile(source, destination);
        await chmod(destination, 0o600);
      }
      if (reference) {
        const extension = reference.name.toLowerCase().endsWith(".nii.gz") ? ".nii.gz" : ".nii";
        const destination = join(inputDirectory, `research_reference${extension}`);
        await copyFile(join(sourceDirectory, reference.name), destination);
        await chmod(destination, 0o600);
      }
      const validation = await runJsonProcess(python, [runner, inputDirectory, "--validate-only"], {
        cwd: repoRoot, timeoutMs: 90_000, spawnImpl: options.spawnImpl
      });
      if (validation.schema_version !== "research-study-validation/v1" || validation.status !== "pass") {
        throw new Error("The demo did not pass the file check");
      }
      const now = new Date();
      const job = {
        jobId, state: "validated", createdAt: now.toISOString(), updatedAt: now.toISOString(),
        expiresAt: new Date(now.getTime() + RETENTION_MS).toISOString(), directory, validation,
        uploadMetadata: {}, evaluationSampleScope: evaluation ? evaluationDemoScope : null,
        result: null, explanation: null, error: null
      };
      jobs.set(jobId, job);
      await persist(job);
      sendJson(res, 201, publicJob(job));
    } catch (error) {
      await rm(directory, { recursive: true, force: true });
      throw error;
    }
  }

  async function runJob(job) {
    activeJobId = job.jobId;
    job.state = "running";
    job.updatedAt = new Date().toISOString();
    await persist(job);
    const inputDirectory = join(job.directory, "input");
    const outputDirectory = join(job.directory, "artifacts");
    try {
      const args = [
        runner, inputDirectory, outputDirectory,
        "--job-id", job.jobId,
        "--checkpoint", checkpoint,
        "--expected-checkpoint-sha256", expectedCheckpointSha256
      ];
      if (process.env.BRAIN_MRI_LLM_MODEL) args.push("--ollama-model", process.env.BRAIN_MRI_LLM_MODEL);
      const result = await runJsonProcess(python, args, {
        cwd: repoRoot, timeoutMs: 30 * 60 * 1000, spawnImpl: options.spawnImpl
      });
      const explanation = JSON.parse(await readFile(join(outputDirectory, "explanation.json"), "utf8"));
      job.result = result;
      job.explanation = explanation;
      job.state = "succeeded";
      job.error = null;
    } catch (error) {
      job.state = "failed";
      job.error = safeProcessError(error.stderr || error.message);
      await rm(outputDirectory, { recursive: true, force: true });
    } finally {
      await rm(inputDirectory, { recursive: true, force: true });
      job.updatedAt = new Date().toISOString();
      await persist(job);
      activeJobId = null;
    }
  }

  async function beginInference(req, res, jobId) {
    validateMutationRequest(req, bindHost, publicHosts);
    rateLimit();
    const job = getJob(jobId);
    if (!job) return sendJson(res, 404, { error: "study_not_found" });
    await refreshCapabilities();
    if (capabilities.inference.status !== "ready") return sendJson(res, 503, { error: "model_unavailable" });
    if (activeJobId) return sendJson(res, 409, { error: "gpu_busy", activeJobId });
    if (job.state !== "validated") return sendJson(res, 409, { error: "study_not_ready", state: job.state });
    void runJob(job);
    sendJson(res, 202, publicJob({ ...job, state: "running", updatedAt: new Date().toISOString() }));
  }

  async function serveArtifact(res, jobId, kind) {
    const job = getJob(jobId);
    const artifact = ARTIFACTS[kind];
    if (!job || job.state !== "succeeded" || !artifact) return sendJson(res, 404, { error: "artifact_not_found" });
    const path = join(job.directory, "artifacts", artifact.file);
    const info = await stat(path);
    res.writeHead(200, {
      "Content-Type": artifact.type,
      "Content-Length": info.size,
      "Content-Disposition": `attachment; filename="${artifact.file}"`,
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff"
    });
    const stream = createReadStream(path);
    stream.on("error", () => res.destroy());
    stream.pipe(res);
  }

  async function clear(req, res, jobId) {
    validateMutationRequest(req, bindHost, publicHosts);
    rateLimit();
    const job = getJob(jobId);
    if (!job) return sendJson(res, 404, { error: "study_not_found" });
    if (job.state === "running") return sendJson(res, 409, { error: "study_running" });
    jobs.delete(jobId);
    await rm(job.directory, { recursive: true, force: true });
    res.writeHead(204, { "Cache-Control": "no-store" }).end();
  }

  async function handle(req, res) {
    await initialize();
    const path = new URL(req.url || "/", "http://localhost").pathname;
    if (req.method === "GET" && path === "/api/capabilities") {
      await refreshCapabilities();
      return sendJson(res, 200, capabilities);
    }
    if (req.method === "POST" && path === "/api/studies/demo") return await loadDemo(req, res);
    if (req.method === "POST" && path === "/api/studies/demo-evaluation") return await loadDemo(req, res, { evaluation: true });
    if (req.method === "POST" && path === "/api/studies") return await upload(req, res);
    const inference = path.match(/^\/api\/studies\/([0-9a-f-]+)\/inference$/);
    if (req.method === "POST" && inference && UUID_V4.test(inference[1])) return await beginInference(req, res, inference[1]);
    const artifact = path.match(/^\/api\/studies\/([0-9a-f-]+)\/artifacts\/(segmentation|receipt|explanation)$/);
    if (req.method === "GET" && artifact && UUID_V4.test(artifact[1])) return await serveArtifact(res, artifact[1], artifact[2]);
    const study = path.match(/^\/api\/studies\/([0-9a-f-]+)$/);
    if (study && UUID_V4.test(study[1])) {
      if (req.method === "GET") {
        const job = getJob(study[1]);
        return job ? sendJson(res, 200, publicJob(job)) : sendJson(res, 404, { error: "study_not_found" });
      }
      if (req.method === "DELETE") return await clear(req, res, study[1]);
    }
    return sendJson(res, 404, { error: "not_found" });
  }

  return {
    handle: async (req, res) => {
      try { await handle(req, res); }
      catch (error) {
        const status = Number(error.status) || 500;
        if (status >= 500) console.error("Study gateway error:", error);
        sendJson(res, status, {
          error: status >= 500 ? "study_gateway_error" : "study_request_rejected",
          detail: status >= 500 ? "The local study gateway could not complete the request." : String(error.message).slice(0, 280)
        });
      }
    },
    initialize,
    jobs,
    capabilities: () => capabilities,
    constants: { EXPECTED_CHECKPOINT_SHA256: expectedCheckpointSha256, MAX_FILE_BYTES, MAX_TOTAL_BYTES }
  };
}

export const studyService = createStudyService();

export function isStudyApiPath(url = "") {
  const path = new URL(url, "http://localhost").pathname;
  return path === "/api/capabilities" || path === "/api/studies" || path.startsWith("/api/studies/");
}
