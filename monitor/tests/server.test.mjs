import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";
import { HOSTS, etagFor, parseCollectorOutput, parseStrictJson, publicSnapshot, requestHandler, resolveStaticPath, runCollector, validateIdentity, validateResource, validateTarget } from "../server.mjs";

test("SSH targets are constrained to user and host", () => {
  assert.equal(validateTarget("software@100.64.0.1"), "software@100.64.0.1");
  assert.equal(HOSTS.nvidia.target, process.env.MONITOR_NVIDIA_TARGET || "software@100.64.0.1");
  assert.equal(HOSTS.amd.target, process.env.MONITOR_AMD_TARGET || "theaa@100.64.0.3");
  assert.throws(() => validateTarget("-oProxyCommand=bad"));
  assert.throws(() => validateTarget("user@host; touch nope"));
  assert.equal(validateIdentity("/home/software/.ssh/brain_mri_monitor_ed25519"), "/home/software/.ssh/brain_mri_monitor_ed25519");
  assert.throws(() => validateIdentity("relative/key"));
});

test("ETags are deterministic and content-sensitive", () => {
  assert.equal(etagFor("same"), etagFor("same"));
  assert.notEqual(etagFor("same"), etagFor("different"));
});

test("strict JSON rejects duplicate keys and trailing content", () => {
  assert.deepEqual(parseStrictJson('{"outer":{"value":1},"items":[true,null]}'), { outer: { value: 1 }, items: [true, null] });
  assert.throws(() => parseStrictJson('{"value":1,"value":2}'), /duplicate JSON key/);
  assert.throws(() => parseStrictJson('{"value":1} false'), /trailing JSON content/);
});

test("collector parsing tolerates a login banner but trusts only the final JSON line", () => {
  assert.deepEqual(parseCollectorOutput("login banner\n{\"id\":\"nvidia\"}\n"), { id: "nvidia" });
  assert.throws(() => parseCollectorOutput("login banner only"));
});

test("static resolution cannot escape the dist root", () => {
  const root = "/tmp/brain-monitor-dist";
  for (const path of ["/", "/assets/app.js", "/../../server.mjs", "/%2e%2e/%2e%2e/etc/passwd"]) {
    const result = resolveStaticPath(path, root);
    assert.ok(result === root || result?.startsWith(`${root}/`));
  }
});

test("a failed host keeps last-good data and becomes stale", () => {
  const source = {
    updatedAt: "2026-08-17T00:01:00Z",
    hosts: {
      nvidia: { reachable: false, data: { id: "nvidia" }, lastSuccessAt: "2026-08-17T00:00:00Z", staleSince: "2026-08-17T00:00:10Z", latencyMs: 10, error: "timeout", nextAttemptAt: "2026-08-17T00:02:00Z" },
      amd: { reachable: true, data: { id: "amd" }, lastSuccessAt: new Date().toISOString(), staleSince: null, latencyMs: 12, error: null, nextAttemptAt: new Date().toISOString() }
    }
  };
  const result = publicSnapshot(source);
  assert.equal(result.hosts.nvidia.stale, true);
  assert.deepEqual(result.hosts.nvidia.data, { id: "nvidia" });
  assert.equal(result.hosts.amd.stale, false);
});

test("collector rejects malformed remote JSON and never invokes a shell", async () => {
  let invocation;
  const spawnImpl = (file, args, options) => {
    invocation = { file, args, options };
    const child = new EventEmitter();
    child.stdout = new PassThrough();
    child.stderr = new PassThrough();
    child.stdin = new PassThrough();
    child.kill = () => {};
    child.stdin.on("finish", () => {
      child.stdout.end("not json");
      queueMicrotask(() => child.emit("close", 0));
    });
    return child;
  };
  await assert.rejects(runCollector("nvidia", { target: "software@100.64.0.1", collector: new URL("../collectors/nvidia.py", import.meta.url).pathname }, { spawnImpl, timeoutMs: 1000 }), /invalid JSON/);
  assert.equal(invocation.file, "ssh");
  assert.equal(invocation.options.stdio[0], "pipe");
  assert.ok(invocation.args.includes("StrictHostKeyChecking=yes"));
  assert.ok(invocation.args.includes("--"));
});

test("fixture resources validate as blocked, deterministic, and non-executing", async () => {
  const { readFile } = await import("node:fs/promises");
  const { join, dirname } = await import("node:path");
  const root = dirname(new URL(import.meta.url).pathname);
  for (const kind of ["evidence", "explanation", "proposals"]) {
    const payload = JSON.parse(await readFile(join(root, "..", "fixtures", `${kind}-fresh.json`), "utf8"));
    const validated = validateResource(kind, payload);
    assert.equal(validated.source.status, "fresh");
    if (kind === "evidence") assert.equal(validated.data.promotion.status, "blocked");
    if (kind === "proposals") assert.ok(validated.data.proposals.every(item => item.executed === false));
  }
});

test("resource validation rejects unsupported versions, bad hashes, and promotion claims", () => {
  assert.throws(() => validateResource("evidence", { schemaVersion: "research-evidence/v9" }), /unsupported/);
  assert.throws(() => validateResource("evidence", { schemaVersion: "research-evidence/v1", generatedAt: "2026-01-01T00:00:00Z", source: { status: "fresh" }, artifactDigest: "bad", data: null }), /digest/);
});

test("default evidence endpoint serves completed rejected-candidate evidence", () => {
  const previous = process.env.MONITOR_FIXTURE_SCENARIO;
  delete process.env.MONITOR_FIXTURE_SCENARIO;
  const chunks = [];
  const response = {
    writeHead(status, headers) { this.status = status; this.headers = headers; return this; },
    end(body) { chunks.push(body); }
  };
  requestHandler({ method: "GET", url: "/api/evidence", headers: {} }, response);
  assert.equal(response.status, 200);
  const body = JSON.parse(chunks.join(""));
  assert.equal(body.data.promotion.status, "rejected");
  assert.equal(body.data.selectedModel.variantId, "baseline");
  assert.equal(body.data.baseline.seedMetrics.length, 3);
  assert.equal(body.data.candidates[0].seedMetrics.length, 3);
  if (previous === undefined) delete process.env.MONITOR_FIXTURE_SCENARIO;
  else process.env.MONITOR_FIXTURE_SCENARIO = previous;
});

test("missing fixture scenarios become a safe contract error", async () => {
  const previous = process.env.MONITOR_FIXTURE_SCENARIO;
  process.env.MONITOR_FIXTURE_SCENARIO = "does-not-exist";
  const chunks = [];
  const response = {
    writeHead(status, headers) { this.status = status; this.headers = headers; return this; },
    end(body) { chunks.push(body); }
  };
  requestHandler({ method: "GET", url: "/api/evidence", headers: {} }, response);
  assert.equal(response.status, 200);
  const body = JSON.parse(chunks.join(""));
  assert.equal(body.source.status, "rejected");
  process.env.MONITOR_FIXTURE_SCENARIO = previous;
});

test("offline stale, outage, rejection, and proposal-state fixtures are contract-valid", async () => {
  const { readFile } = await import("node:fs/promises");
  const { join, dirname } = await import("node:path");
  const root = dirname(new URL(import.meta.url).pathname);
  for (const [kind, scenarios] of Object.entries({ evidence: ["stale", "unavailable", "outage", "rejected"], explanation: ["unavailable", "rejected"], proposals: ["unsafe", "already-complete"] })) {
    for (const scenario of scenarios) {
      const payload = JSON.parse(await readFile(join(root, "..", "fixtures", `${kind}-${scenario}.json`), "utf8"));
      const result = validateResource(kind, payload);
      assert.ok(["stale", "unavailable", "rejected", "fresh"].includes(result.source.status));
      if (kind === "proposals") assert.ok(result.data.proposals.every(item => item.executed === false));
    }
  }
});
