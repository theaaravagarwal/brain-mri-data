# Prototype acceptance — September 4, 2026

Scope: repeated sample operational checks of the frozen prototype. No new cohort,
model selection, tuning, or training occurred.

- Pipeline sample passed via `http://100.64.0.1:4173` at 02:35:09 UTC September 5.
  Output contained 15,813 voxels; explanation rendering was validated.
- Reference sample passed after deployment at 02:36:57 UTC September 5.
  Dice: 0.9537691550967009; HD95: 3 mm; output: 196,020 voxels.
- Both commands checked checkpoint identity, segmentation bytes against the
  receipt, geometry preservation, the pinned PDF hash, and explanation downloads.
- Both temporary jobs were deleted, followed by verified HTTP 404 responses.
- The first check immediately following the service restart received HTTP 502;
  after service startup the complete check passed. The checker fails visibly
  rather than reporting success while the application is unavailable.
- Storage failure recovery is covered by a regression that removes one temporary
  job directory and verifies the next study still completes.
- Local suite: 27 service tests and 13 UI tests passed; TypeScript/Vite build passed.
- Browser verification passed: sample selection, geometry validation, inference,
  result rendering, technical receipt download, and clearing the result. Browser
  console reported zero errors and warnings.
- `.7` prototype and Ollama services and `.1` proxy are enabled. User lingering
  is enabled on both hosts. Reboot recovery was configured, not reboot-tested.

Reproduce with `scripts/check_prototype.mjs`; see `docs/prototype-operations.md`.
