# MRI viewer delivery — September 4, 2026

## Deployed scope

Live application: http://100.64.0.1:4173, proxied to `.7`.
Active application commit: `daba1b782a431dc8c8ab7367e2a86d7f32b45d81`.
No training was started; `.3` was not required for this delivery.

The app now provides four-modality, three-plane viewing, adjustable overlays,
slice controls, jump-to-outline, and synchronized expert/model comparison.
Niivue is pinned at 0.69.0 and loaded only for completed results. Advanced
provenance is collapsed; Dice and HD95 have plain-language definitions.

Completed studies can be reopened from this browser until their 24-hour expiry.
Each study requires its random bearer capability; the server stores its hash,
not the token. Clearing browser storage loses access to retained studies.
Sanitized scan copies remain on `.7` for viewing; original uploads are removed
after inference. Sanitization removes header text, not identifying anatomy:
these are still sensitive imaging data, not guaranteed anonymous images.
All rendering stays in the browser; the LLM receives validated metadata only.

The result ZIP contains the segmentation, receipt, explanation, frozen validation
PDF, and SHA256SUMS. It deliberately excludes scan volumes.

## Verification

- 30 service tests and 13 UI tests passed; TypeScript/Vite build passed.
- Python runner/language checks passed, including nontrivial affine geometry,
  preserved voxel values, sanitized headers, and outline jump coordinates.
- Live regular sample: 15,813 output voxels; validated local LLM explanation.
- Live reference sample: Dice 0.9537691550967009; HD95 3 mm; 196,020 output voxels.
- These are repeated smoke tests, **not new independent validation evidence**.
- Downloaded ZIP integrity and every SHA256SUMS entry were checked.
- Protected routes without a token returned 404; a second browser had no history.
- Concurrent submission, active-job deletion protection, expiry cleanup, and
  persisted completed-job restoration are covered by service tests.
- A real retained result reopened after deployment restarted the service.
- Desktop comparison and mobile axial/reference rendering were inspected.
  Browser console: zero errors. Resource origins: same origin only.
- Local screenshots: `output/playwright/mri-viewer-desktop.png` and
  `output/playwright/mri-viewer-mobile.png` (not published research data).

Checkpoint remains
`121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5`.
Frozen validation PDF remains
`f9aa0f56ce129059a47816826a12a074794027f9dc8b00af38d4acf921623eef`.
Segmentation file hashes differ from the earlier app because private header text
is now stripped; checkpoint and sample voxel results did not change.

## Operations and remaining work

Deploy: `bash scripts/deploy_prototype.sh deploy <commit>`.
Rollback: `bash scripts/deploy_prototype.sh rollback`.
Releases use separate directories and preserve the shared expiring job store.
Deployment runs both live sample checks and rolls back on acceptance failure.
See `docs/prototype-operations.md` for the complete operating contract.

Full reboot recovery remains unverified: `.7` rejected the noninteractive reboot
with `Interactive authentication required`. No host was rebooted. An operator
must authenticate to reboot `.7`, verify app/Ollama recovery and retained-result
access, then reboot `.1` and verify the proxy. Service restart recovery was tested.

The next quality step needs a genuinely untouched cohort and qualified overlay
review, particularly small-lesion misses. No new cohort was available here.
Do not tune on the existing 60-case external cohort or claim clinical readiness.
