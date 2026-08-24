# Brain MRI training monitor

A local, read-only dashboard for the NVIDIA and AMD workers. It binds to
`127.0.0.1`, reads fixed host telemetry and run artifacts over Tailscale SSH,
and never exposes queue, pause, restart, or shell controls.

## Start

Prerequisites: Node.js 26+, `ssh`, Tailscale connectivity, and accepted host
keys for `theaa@100.64.0.3` and `b@100.64.0.5`.

```bash
cd monitor
npm install
npm start
```

Open <http://127.0.0.1:4173>. `npm start` builds once and serves the production
bundle. For frontend development, run `node server.mjs` in one terminal and
`npm run dev` in another, then open <http://127.0.0.1:5173>.

## Network behavior

- Hosts are collected concurrently every 10 seconds.
- SSH uses compression and a persistent multiplexed connection under a
  mode-700, user-specific `/tmp/brain-mri-monitor-<uid>/` directory, reducing
  repeated handshakes over unstable Wi-Fi while staying below macOS socket
  path limits.
- Each attempt has a hard timeout. Failures back off up to 60 seconds.
- Last-good values stay visible with their confirmation age and error.
- Browser requests use ETags, so unchanged local snapshots return no body.

Override a fixed target only from the server environment:

```bash
MONITOR_NVIDIA_TARGET=theaa@10.0.0.65 npm start
```

Targets must remain `user@host` values. There are no browser-provided host,
path, or command parameters.

## Offline UI fixture

```bash
MONITOR_FIXTURE=1 npm start
```

This serves representative non-identifying data without contacting either
worker. Run `npm test` for gateway safety checks and a production build.
