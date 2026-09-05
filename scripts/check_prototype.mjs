#!/usr/bin/env node
// Runs inference on a built-in sample, verifies downloads, and deletes only its own job.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { setTimeout as delay } from "node:timers/promises";

const base = new URL(process.argv[2] || "http://100.64.0.1:4173");
const evaluation = process.argv.includes("--evaluation");
const requireLlm = process.argv.includes("--require-llm");
const checkpoint = "121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5";
const reportHash = "f9aa0f56ce129059a47816826a12a074794027f9dc8b00af38d4acf921623eef";
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
let jobId;
let accessToken;

async function request(path, method = "GET", expected = 200) {
  const response = await fetch(new URL(path, base), {
    method, headers: { Origin: base.origin, ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}) }, signal: AbortSignal.timeout(60_000)
  });
  assert.equal(response.status, expected, `${method} ${path}: ${response.status}`);
  return response;
}

try {
  const caps = await (await request("/api/capabilities")).json();
  assert.equal(caps.inference.status, "ready");
  assert.equal(caps.inference.observedCheckpointSha256, checkpoint);
  assert.equal(caps.externalBenchmark.case_count, 60);
  const report = Buffer.from(await (await request("/api/external-benchmark/report")).arrayBuffer());
  assert.equal(hash(report), reportHash, "Validation report changed");
  const route = evaluation ? "demo-evaluation" : "demo";
  let job = await (await request(`/api/studies/${route}`, "POST", 201)).json();
  jobId = job.jobId;
  accessToken = job.accessToken;
  assert.equal(job.state, "validated");
  await request(`/api/studies/${jobId}/inference`, "POST", 202);
  const deadline = Date.now() + 31 * 60_000;
  do {
    await delay(2000);
    job = await (await request(`/api/studies/${jobId}`)).json();
    if (job.state !== "running") break;
  } while (Date.now() < deadline);
  assert.equal(job.state, "succeeded", `Inference did not complete: ${job.error || job.state}`);
  const receipt = await (await request(`/api/studies/${jobId}/artifacts/receipt`)).json();
  const explanation = await (await request(`/api/studies/${jobId}/artifacts/explanation`)).json();
  const mask = Buffer.from(await (await request(`/api/studies/${jobId}/artifacts/segmentation`)).arrayBuffer());
  assert.equal(receipt.job_id, jobId);
  assert.equal(receipt.model.checkpoint_sha256, checkpoint);
  assert.equal(hash(mask), receipt.segmentation.output_sha256);
  assert.deepEqual(receipt.segmentation, job.result.segmentation);
  assert.equal(receipt.segmentation.geometry_preserved, true);
  assert.ok(job.viewing?.volumes.includes("flair"), "Viewing files missing");
  for (const modality of job.viewing.volumes) {
    const bytes = Buffer.from(await (await request(`/api/studies/${jobId}/viewing/${modality}`)).arrayBuffer());
    assert.equal(hash(bytes), job.viewing.sha256[modality], `Viewing hash mismatch: ${modality}`);
  }
  const denied = await fetch(new URL(`/api/studies/${jobId}`, base));
  assert.equal(denied.status, 404, "Study must require its token");
  const bundle = Buffer.from(await (await request(`/api/studies/${jobId}/package`)).arrayBuffer());
  assert.equal(bundle.subarray(0, 2).toString(), "PK", "Result package must be ZIP");
  assert.ok(explanation.deterministic.summary);
  if (requireLlm) assert.equal(explanation.llm.status, "validated", explanation.llm.reason);
  if (evaluation) {
    assert.equal(receipt.evaluation.status, "complete");
    assert.ok(Number.isFinite(receipt.evaluation.whole_lesion_dice));
  }
  console.log(JSON.stringify({
    status: "passed", checkedAt: new Date().toISOString(), endpoint: base.origin,
    checkpoint, reportSha256: hash(report), segmentationSha256: hash(mask),
    voxels: receipt.segmentation.nonzero_voxels,
    llm: explanation.llm.status, evaluation: receipt.evaluation,
    scope: "repeat sample smoke test; not new independent validation"
  }, null, 2));
} catch (error) {
  console.error(`Acceptance check failed: ${error.message}`);
  process.exitCode = 1;
} finally {
  if (jobId) {
    try {
      await request(`/api/studies/${jobId}`, "DELETE", 204);
      await request(`/api/studies/${jobId}`, "GET", 404);
      console.log("Temporary sample result cleared and deletion verified.");
    } catch (error) {
      console.error(`Sample cleanup incomplete (${jobId}): ${error.message}`);
      process.exitCode = 1;
    }
  }
}
