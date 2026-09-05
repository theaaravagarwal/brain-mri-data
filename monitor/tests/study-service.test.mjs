import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import test from "node:test";
import { createStudyService } from "../study-service.mjs";

function responseCapture() {
  const response = new PassThrough();
  const chunks = [];
  response.status = 0;
  response.headers = {};
  response.chunks = chunks;
  response.writeHead = function writeHead(status, headers = {}) { this.status = status; this.headers = headers; return this; };
  response.on("data", chunk => chunks.push(Buffer.from(chunk)));
  response.complete = new Promise(resolve => response.on("finish", () => resolve(response)));
  return response;
}

async function call(service, method, url, { body, headers = {}, remoteAddress = "127.0.0.1" } = {}) {
  const id = url.match(/\/api\/studies\/([0-9a-f-]{36})/)?.[1];
  const token = service.testTokens?.get(id);
  const req = new PassThrough();
  Object.assign(req, {
    method,
    url,
    headers: { host: "127.0.0.1:4173", ...(token ? { authorization: `Bearer ${token}` } : {}), ...headers },
    socket: { remoteAddress }
  });
  const res = responseCapture();
  const handling = service.handle(req, res);
  req.end(body);
  await handling;
  await res.complete;
  if (res.status === 201) {
    const created = JSON.parse(Buffer.concat(res.chunks).toString());
    service.testTokens ??= new Map();
    service.testTokens.set(created.jobId, created.accessToken);
  }
  return { status: res.status, headers: res.headers, body: Buffer.concat(res.chunks), json: () => JSON.parse(Buffer.concat(res.chunks).toString()) };
}

function multipartStudy(boundary, withReference = false) {
  const parts = [];
  for (const field of ["t1", "t1ce", "t2", "flair"]) {
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="${field}"; filename="private-${field}.nii.gz"\r\nContent-Type: application/gzip\r\n\r\n`));
    parts.push(Buffer.from(`compressed-${field}`));
    parts.push(Buffer.from("\r\n"));
  }
  if (withReference) {
    parts.push(Buffer.from(`--${boundary}\r\nContent-Disposition: form-data; name="reference"; filename="expert-private-name.nii"\r\nContent-Type: application/octet-stream\r\n\r\n`));
    parts.push(Buffer.from("reference-mask"));
    parts.push(Buffer.from("\r\n"));
  }
  parts.push(Buffer.from(`--${boundary}--\r\n`));
  return Buffer.concat(parts);
}

function fakeSpawn(inferenceDelay = 0) {
  return (_command, args) => {
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.kill = () => {};
    setTimeout(() => {
      const input = args[1];
      const hasReference = existsSync(join(input, "research_reference.nii")) || existsSync(join(input, "research_reference.nii.gz"));
      if (args.includes("--validate-only")) {
        child.stdout.end(JSON.stringify({
          schema_version: "research-study-validation/v1",
          status: "pass",
          modality_count: 4,
          modalities: ["t1", "t1ce", "t2", "flair"],
          geometry_match: true,
          shape: [8, 9, 10],
          spacing_mm: [1, 1, 1],
          geometry_sha256: "a".repeat(64),
          modality_sha256: { t1: "b".repeat(64), t1ce: "c".repeat(64), t2: "d".repeat(64), flair: "e".repeat(64) },
          ...(hasReference ? { reference_mask: { status: "pass", geometry_match: true, sha256: "9".repeat(64), labels: [0, 1, 4], nonzero_voxels: 10 } } : {})
        }));
      } else {
        const output = args[2];
        mkdirSync(output, { recursive: true });
        writeFileSync(join(output, "research_segmentation.nii.gz"), "mask");
        writeFileSync(join(output, "receipt.json"), "{}\n");
        writeFileSync(join(output, "explanation.json"), JSON.stringify({
          schema_version: "research-segmentation-explanation/v1",
          deterministic: { disclaimer: "Research only", summary: "Complete", evidence: [], limitations: "Research only", abstained: false },
          llm: { status: "unavailable", artifact: null, reason: "not configured", model_name: null, model_digest: null }
        }));
        child.stdout.end(JSON.stringify({
          schema_version: "research-segmentation-result/v1",
          disclaimer: "Research only",
          segmentation: { nonzero_voxels: 12, output_sha256: "f".repeat(64), output_shape: [8, 9, 10], geometry_preserved: true, labels: [0, 1], label_count: 2, status: "complete" },
          evaluation: hasReference ? { status: "complete", scope: "single_user_supplied_reference", whole_lesion_dice: 0.8, whole_lesion_iou: 0.667, precision: 0.75, recall: 0.86, hd95_mm: 2.1, true_positive_voxels: 6, false_positive_voxels: 2, false_negative_voxels: 1 } : null,
          provenance: { model_id: "glioma-segresnet-20260828", checkpoint_sha256: "0".repeat(64) }
        }));
      }
      queueMicrotask(() => child.emit("close", 0));
    }, args.includes("--validate-only") ? 0 : inferenceDelay);
    return child;
  };
}

async function fixture({ bindHost, publicHosts, deviceProbe = async () => true, inferenceDelay = 0 } = {}) {
  const root = await mkdtemp(join(tmpdir(), "brain-study-service-"));
  const repoRoot = join(root, "repo");
  const runtimeRoot = join(root, "runtime");
  const python = join(repoRoot, ".venv", "bin", "python");
  const runner = join(repoRoot, "scripts", "runner.py");
  const checkpoint = join(repoRoot, "runs", "best.pt");
  const demoDirectory = join(repoRoot, "demo");
  const evaluationDemoDirectory = join(repoRoot, "external-demo");
  const benchmarkDirectory = join(repoRoot, "artifacts", "fixed-segresnet-external");
  await mkdir(join(repoRoot, ".venv", "bin"), { recursive: true });
  await mkdir(join(repoRoot, "scripts"), { recursive: true });
  await mkdir(join(repoRoot, "runs"), { recursive: true });
  await mkdir(demoDirectory, { recursive: true });
  await mkdir(evaluationDemoDirectory, { recursive: true });
  await mkdir(benchmarkDirectory, { recursive: true });
  for (const modality of ["t1", "t1ce", "t2", "flair"]) await writeFile(join(demoDirectory, `sample_${modality}.nii`), modality);
  await writeFile(join(demoDirectory, "sample_seg.nii"), "reference");
  for (const suffix of ["t1n", "t1c", "t2w", "t2f"]) await writeFile(join(evaluationDemoDirectory, `external-${suffix}.nii.gz`), suffix);
  await writeFile(join(evaluationDemoDirectory, "external-seg.nii.gz"), "reference");
  await writeFile(python, "python");
  await writeFile(runner, "runner");
  await writeFile(checkpoint, "checkpoint");
  const expectedCheckpointSha256 = createHash("sha256").update("checkpoint").digest("hex");
  const serviceOptions = { repoRoot, runtimeRoot, python, runner, checkpoint, expectedCheckpointSha256, demoDirectory, evaluationDemoDirectory, evaluationDemoScope: "external_public", benchmarkDirectory, spawnImpl: fakeSpawn(inferenceDelay), deviceProbe, bindHost, publicHosts };
  const service = createStudyService(serviceOptions);
  await service.initialize();
  return { root, runtimeRoot, checkpoint, benchmarkDirectory, service, serviceOptions };
}

test("a job persistence failure releases the GPU slot for the next study", async () => {
  const { service, runtimeRoot } = await fixture();
  const first = (await call(service, "POST", "/api/studies/demo")).json();
  await rm(join(runtimeRoot, first.jobId), { recursive: true });
  assert.equal((await call(service, "POST", `/api/studies/${first.jobId}/inference`)).status, 202);
  // Let the failed write and final cleanup complete before starting another job.
  await new Promise(resolve => setTimeout(resolve, 50));
  assert.equal((await call(service, "GET", `/api/studies/${first.jobId}`)).json().state, "failed");
  const second = (await call(service, "POST", "/api/studies/demo")).json();
  assert.equal((await call(service, "POST", `/api/studies/${second.jobId}/inference`)).status, 202);
  await new Promise(resolve => setTimeout(resolve, 50));
  assert.equal((await call(service, "GET", `/api/studies/${second.jobId}`)).json().state, "succeeded");
});

test("capabilities expose the exact ready checkpoint without a path", async () => {
  const { service } = await fixture();
  const response = await call(service, "GET", "/api/capabilities");
  assert.equal(response.status, 200);
  const value = response.json();
  assert.equal(value.inference.status, "ready");
  assert.equal(value.inference.checkpointSha256, service.constants.EXPECTED_CHECKPOINT_SHA256);
  assert.equal(value.demoAvailable, true);
  assert.equal(value.evaluationDemoAvailable, true);
  assert.equal(JSON.stringify(value).includes("/runs/"), false);
});

test("study tokens protect status, inference, files, package and deletion", async () => {
  const { service, runtimeRoot } = await fixture();
  const created = (await call(service, "POST", "/api/studies/demo")).json();
  assert.equal(created.accessToken.length, 64);
  const stored = await readFile(join(runtimeRoot, created.jobId, "job.json"), "utf8");
  assert.equal(stored.includes(created.accessToken), false);
  for (const [method, suffix] of [["GET", ""], ["POST", "/inference"], ["GET", "/viewing/flair"], ["GET", "/package"], ["DELETE", ""]]) {
    const response = await call(service, method, `/api/studies/${created.jobId}${suffix}`, { headers: { authorization: "Bearer wrong" } });
    assert.equal(response.status, 404);
  }
  const response = await call(service, "GET", `/api/studies/${created.jobId}`);
  assert.equal(response.status, 200);
  assert.equal(response.json().accessToken, undefined);
  assert.equal(response.json().accessHash, undefined);
});

test("simultaneous submissions keep the second study available and reject clearing an active job", async () => {
  const { service } = await fixture({ inferenceDelay: 200 });
  const first = (await call(service, "POST", "/api/studies/demo")).json();
  const second = (await call(service, "POST", "/api/studies/demo")).json();
  const started = await Promise.all([first, second].map(job => call(service, "POST", `/api/studies/${job.jobId}/inference`)));
  assert.deepEqual(started.map(value => value.status).sort(), [202, 409]);
  const running = started[0].status === 202 ? first : second;
  const waiting = running === first ? second : first;
  assert.equal((await call(service, "DELETE", `/api/studies/${running.jobId}`)).status, 409);
  assert.equal((await call(service, "GET", `/api/studies/${waiting.jobId}`)).json().state, "validated");
  await new Promise(resolve => setTimeout(resolve, 250));
});

test("completed studies retain access after restart and expire with their files", async () => {
  const { service, serviceOptions, runtimeRoot } = await fixture();
  const job = (await call(service, "POST", "/api/studies/demo")).json();
  await call(service, "POST", `/api/studies/${job.jobId}/inference`);
  await new Promise(resolve => setTimeout(resolve, 30));
  const restarted = createStudyService(serviceOptions);
  const headers = { authorization: `Bearer ${job.accessToken}` };
  assert.equal((await call(restarted, "GET", `/api/studies/${job.jobId}`, { headers })).json().state, "succeeded");
  restarted.jobs.get(job.jobId).expiresAt = "2000-01-01T00:00:00Z";
  assert.equal((await call(restarted, "GET", `/api/studies/${job.jobId}`, { headers })).status, 404);
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(existsSync(join(runtimeRoot, job.jobId)), false);
});

test("capabilities expose only aggregate external benchmark results", async () => {
  const { service, benchmarkDirectory } = await fixture();
  await writeFile(join(benchmarkDirectory, "summary.public.json"), JSON.stringify({
    schema_version: "fixed-segresnet-external-summary/v1",
    benchmark_id: "fixed-test",
    status: "complete",
    case_count: 60,
    metrics: { whole_lesion_dice: { mean: 0.9, median: 0.91, mean_ci95: [0.88, 0.92] } }
  }));
  const response = await call(service, "GET", "/api/capabilities");
  assert.equal(response.status, 200);
  assert.equal(response.json().externalBenchmark.case_count, 60);
  assert.equal(JSON.stringify(response.json()).includes("case_token"), false);
});

test("external validation report downloads as a PDF", async () => {
  const { service, benchmarkDirectory } = await fixture();
  await writeFile(join(benchmarkDirectory, "validation-report.pdf"), "%PDF-release-report");
  const response = await call(service, "GET", "/api/external-benchmark/report");
  assert.equal(response.status, 200);
  assert.equal(response.headers["Content-Type"], "application/pdf");
  assert.match(response.headers["Content-Disposition"], /fixed-segresnet-external-validation-report\.pdf/);
  assert.equal(response.body.toString(), "%PDF-release-report");
});

test("built-in demo creates a validated private study without an upload", async () => {
  const { service, runtimeRoot } = await fixture();
  const response = await call(service, "POST", "/api/studies/demo", {
    headers: { origin: "http://127.0.0.1:4173", "sec-fetch-site": "same-origin" }
  });
  assert.equal(response.status, 201, response.body.toString());
  const job = response.json();
  assert.equal(job.state, "validated");
  assert.equal(JSON.stringify(job).includes("sample_"), false);
  assert.deepEqual((await readdir(join(runtimeRoot, job.jobId, "input"))).sort(), [
    "research_input_0000.nii", "research_input_0001.nii", "research_input_0002.nii", "research_input_0003.nii"
  ]);
});

test("built-in accuracy sample includes a private reference mask", async () => {
  const { service, runtimeRoot } = await fixture();
  const response = await call(service, "POST", "/api/studies/demo-evaluation", {
    headers: { origin: "http://127.0.0.1:4173", "sec-fetch-site": "same-origin" }
  });
  assert.equal(response.status, 201, response.body.toString());
  const job = response.json();
  assert.equal(job.validation.reference_mask.nonzero_voxels, 10);
  assert.equal(job.evaluationSampleScope, "external_public");
  assert.equal(JSON.stringify(job).includes("sample_seg"), false);
  assert.ok((await readdir(join(runtimeRoot, job.jobId, "input"))).includes("research_reference.nii.gz"));
});

test("capabilities fail closed when CUDA is unavailable", async () => {
  const { service } = await fixture({ deviceProbe: async () => false });
  const response = await call(service, "GET", "/api/capabilities");
  assert.equal(response.status, 200);
  assert.equal(response.json().inference.status, "unavailable");
});

test("cross-origin study uploads are rejected", async () => {
  const { service } = await fixture();
  const response = await call(service, "POST", "/api/studies", {
    headers: { origin: "https://attacker.example", "content-type": "multipart/form-data; boundary=nope" },
    body: Buffer.from("--nope--\r\n")
  });
  assert.equal(response.status, 403);
});

test("an exact Tailscale bind accepts tailnet clients and rejects other networks", async () => {
  const { service } = await fixture({ bindHost: "100.64.0.1" });
  const headers = { host: "100.64.0.1:4173", "content-type": "multipart/form-data; boundary=nope" };
  const tailnet = await call(service, "POST", "/api/studies", {
    headers,
    remoteAddress: "100.88.1.2",
    body: Buffer.from("--nope--\r\n")
  });
  assert.equal(tailnet.status, 400);
  const lan = await call(service, "POST", "/api/studies", {
    headers,
    remoteAddress: "192.168.1.20",
    body: Buffer.from("--nope--\r\n")
  });
  assert.equal(lan.status, 403);
});

test("a fixed Tailscale proxy host can be allowlisted without accepting arbitrary hosts", async () => {
  const { service } = await fixture({ bindHost: "100.64.0.7", publicHosts: ["100.64.0.1", "100.64.0.7"] });
  const common = { "content-type": "multipart/form-data; boundary=nope" };
  const forwarded = await call(service, "POST", "/api/studies", {
    headers: { ...common, host: "100.64.0.1:4173", origin: "http://100.64.0.1:4173", "sec-fetch-site": "same-origin" },
    remoteAddress: "100.64.0.1",
    body: Buffer.from("--nope--\r\n")
  });
  assert.equal(forwarded.status, 400);
  const attacker = await call(service, "POST", "/api/studies", {
    headers: { ...common, host: "100.64.0.99:4173" },
    remoteAddress: "100.64.0.1",
    body: Buffer.from("--nope--\r\n")
  });
  assert.equal(attacker.status, 403);
});

test("a changed checkpoint rejects the study before upload or validation", async () => {
  const { checkpoint, runtimeRoot, service } = await fixture();
  await writeFile(checkpoint, "tampered checkpoint");
  const boundary = "brain-research-unavailable";
  const response = await call(service, "POST", "/api/studies", {
    headers: {
      origin: "http://127.0.0.1:4173",
      "sec-fetch-site": "same-origin",
      "content-type": `multipart/form-data; boundary=${boundary}`
    },
    body: multipartStudy(boundary)
  });
  assert.equal(response.status, 503);
  assert.match(response.json().detail, /not uploaded/i);
  assert.equal(service.jobs.size, 0);
  assert.deepEqual(await readdir(runtimeRoot), []);
});

test("four-volume upload validates, runs once, downloads artifacts, and clears", async () => {
  const { service, runtimeRoot } = await fixture();
  const boundary = "brain-research-boundary";
  const upload = await call(service, "POST", "/api/studies", {
    headers: {
      origin: "http://127.0.0.1:4173",
      "sec-fetch-site": "same-origin",
      "content-type": `multipart/form-data; boundary=${boundary}`
    },
    body: multipartStudy(boundary)
  });
  assert.equal(upload.status, 201, upload.body.toString());
  const validated = upload.json();
  assert.equal(validated.state, "validated");
  assert.equal(JSON.stringify(validated).includes("private-t1"), false);

  const started = await call(service, "POST", `/api/studies/${validated.jobId}/inference`, {
    headers: { origin: "http://127.0.0.1:4173", "sec-fetch-site": "same-origin" }
  });
  assert.equal(started.status, 202);
  await new Promise(resolve => setTimeout(resolve, 20));
  const status = await call(service, "GET", `/api/studies/${validated.jobId}`);
  assert.equal(status.json().state, "succeeded");
  await assert.rejects(readFile(join(runtimeRoot, validated.jobId, "input", "research_input_0000.nii.gz")));

  const artifact = await call(service, "GET", `/api/studies/${validated.jobId}/artifacts/segmentation`);
  assert.equal(artifact.status, 200);
  assert.equal(artifact.body.toString(), "mask");

  const cleared = await call(service, "DELETE", `/api/studies/${validated.jobId}`, {
    headers: { origin: "http://127.0.0.1:4173", "sec-fetch-site": "same-origin" }
  });
  assert.equal(cleared.status, 204);
  const missing = await call(service, "GET", `/api/studies/${validated.jobId}`);
  assert.equal(missing.status, 404);
});

test("optional reference mask is validated, evaluated, and deleted after processing", async () => {
  const { service, runtimeRoot } = await fixture();
  const boundary = "brain-research-evaluation";
  const upload = await call(service, "POST", "/api/studies", {
    headers: {
      origin: "http://127.0.0.1:4173",
      "sec-fetch-site": "same-origin",
      "content-type": `multipart/form-data; boundary=${boundary}`
    },
    body: multipartStudy(boundary, true)
  });
  assert.equal(upload.status, 201, upload.body.toString());
  const validated = upload.json();
  assert.equal(validated.validation.reference_mask.nonzero_voxels, 10);
  assert.equal(JSON.stringify(validated).includes("expert-private-name"), false);
  assert.deepEqual((await readdir(join(runtimeRoot, validated.jobId, "input"))).sort(), [
    "research_input_0000.nii.gz", "research_input_0001.nii.gz", "research_input_0002.nii.gz", "research_input_0003.nii.gz", "research_reference.nii"
  ]);

  await call(service, "POST", `/api/studies/${validated.jobId}/inference`, {
    headers: { origin: "http://127.0.0.1:4173", "sec-fetch-site": "same-origin" }
  });
  await new Promise(resolve => setTimeout(resolve, 20));
  const status = await call(service, "GET", `/api/studies/${validated.jobId}`);
  assert.equal(status.json().result.evaluation.whole_lesion_dice, 0.8);
  await assert.rejects(readFile(join(runtimeRoot, validated.jobId, "input", "research_reference.nii")));
});
