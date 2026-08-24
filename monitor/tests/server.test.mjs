import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import test from "node:test";
import { etagFor, publicSnapshot, resolveStaticPath, runCollector, validateTarget } from "../server.mjs";

test("SSH targets are constrained to user and host", () => {
  assert.equal(validateTarget("theaa@100.64.0.3"), "theaa@100.64.0.3");
  assert.throws(() => validateTarget("-oProxyCommand=bad"));
  assert.throws(() => validateTarget("user@host; touch nope"));
});

test("ETags are deterministic and content-sensitive", () => {
  assert.equal(etagFor("same"), etagFor("same"));
  assert.notEqual(etagFor("same"), etagFor("different"));
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
  await assert.rejects(runCollector("nvidia", { target: "theaa@100.64.0.3", collector: new URL("../collectors/nvidia.py", import.meta.url).pathname }, { spawnImpl, timeoutMs: 1000 }), /invalid JSON/);
  assert.equal(invocation.file, "ssh");
  assert.equal(invocation.options.stdio[0], "pipe");
  assert.ok(invocation.args.includes("StrictHostKeyChecking=yes"));
  assert.ok(invocation.args.includes("--"));
});
