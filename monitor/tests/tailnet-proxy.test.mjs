import assert from "node:assert/strict";
import test from "node:test";
import { proxyConfig } from "../tailnet-proxy.mjs";

test("tailnet proxy defaults to the stable entrypoint and application host", () => {
  const config = proxyConfig({});
  assert.equal(config.bindHost, "100.64.0.1");
  assert.equal(config.port, 4173);
  assert.equal(config.target.origin, "http://100.64.0.7:4173");
});

test("tailnet proxy rejects public, credentialed, and path targets", () => {
  assert.throws(() => proxyConfig({ BRAIN_MRI_PROXY_TARGET: "https://example.com" }));
  assert.throws(() => proxyConfig({ BRAIN_MRI_PROXY_TARGET: "http://user:secret@100.64.0.7:4173" }));
  assert.throws(() => proxyConfig({ BRAIN_MRI_PROXY_TARGET: "http://100.64.0.7:4173/nested" }));
  assert.throws(() => proxyConfig({ BRAIN_MRI_PROXY_BIND: "0.0.0.0" }));
});
