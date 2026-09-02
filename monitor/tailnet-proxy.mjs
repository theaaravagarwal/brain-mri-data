import { createServer, request as httpRequest } from "node:http";
import { fileURLToPath } from "node:url";

const tailscaleIpv4 = /^100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.(?:\d{1,3})\.(?:\d{1,3})$/;

export function proxyConfig(env = process.env) {
  const bindHost = env.BRAIN_MRI_PROXY_BIND || "100.64.0.1";
  const port = Number(env.BRAIN_MRI_PROXY_PORT || 4173);
  const target = new URL(env.BRAIN_MRI_PROXY_TARGET || "http://100.64.0.7:4173");
  if (!tailscaleIpv4.test(bindHost) || !Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("The proxy must bind to an exact Tailscale IPv4 address and valid port");
  }
  if (target.protocol !== "http:" || !tailscaleIpv4.test(target.hostname) || target.username || target.password || target.pathname !== "/") {
    throw new Error("The proxy target must be an uncredentialed Tailscale HTTP origin");
  }
  return { bindHost, port, target };
}

export function createTailnetProxy(config = proxyConfig()) {
  return createServer((incoming, outgoing) => {
    const upstream = httpRequest({
      protocol: config.target.protocol,
      hostname: config.target.hostname,
      port: config.target.port,
      method: incoming.method,
      path: incoming.url,
      headers: incoming.headers,
      timeout: 35 * 60 * 1000
    }, response => {
      outgoing.writeHead(response.statusCode || 502, response.headers);
      response.pipe(outgoing);
    });
    upstream.on("timeout", () => upstream.destroy(new Error("upstream timeout")));
    upstream.on("error", () => {
      if (!outgoing.headersSent) outgoing.writeHead(502, { "Content-Type": "application/json", "Cache-Control": "no-store" });
      outgoing.end('{"error":"prototype_upstream_unavailable"}');
    });
    incoming.on("aborted", () => upstream.destroy());
    incoming.pipe(upstream);
  });
}

async function main() {
  const config = proxyConfig();
  createTailnetProxy(config).listen(config.port, config.bindHost, () => {
    console.log(`Brain MRI tailnet proxy: http://${config.bindHost}:${config.port} -> ${config.target.origin}`);
  });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error(error);
    process.exitCode = 1;
  });
}
