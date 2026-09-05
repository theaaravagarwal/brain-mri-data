# Prototype operations

## What is finished

The prototype supports four-volume upload and geometry validation, a fixed CNN,
downloadable segmentation and exact provenance, optional reference-mask scoring,
and a local metadata-only LLM explanation. The frozen checkpoint and the 60-case
external benchmark are unchanged. Empty and very small outputs require review.

Release baseline: `prototype-v0.1.0`. This finishing pass adds recovery from job
storage errors and a repeatable acceptance command. It does not train a model.

## Try it

1. Open <http://100.64.0.1:4173> on the tailnet.
2. Use the built-in sample for a quick demonstration. Use the accuracy sample
   for a reference-mask comparison. These are repeat demonstrations, not new tests.
3. Alternatively, select your own T1, T1ce, T2, and FLAIR NIfTI files. Supply an
   expert reference mask if available. Without a reference the app cannot score accuracy.
4. Run the outline and download the segmentation, receipt, and explanation.
5. Clear the result when finished. Artifacts otherwise expire after 24 hours.

## Repeatable acceptance check

Requires Node.js 24 and tailnet access. From the repository root:

```bash
node scripts/check_prototype.mjs http://100.64.0.1:4173 --require-llm
node scripts/check_prototype.mjs http://100.64.0.1:4173 --evaluation --require-llm
```

Run sequentially: the server allows one inference at a time. Each command creates
one sample job, verifies readiness, the pinned report hash, inference, checkpoint
identity, segmentation bytes against the receipt hash, preserved geometry,
explanation, and optional reference metrics. It then deletes only its own job and
verifies that job is unavailable. A failed assertion or cleanup exits nonzero.
On a timeout an actively running job cannot be deleted; the command reports its
identifier. Investigate the service before clearing that job after completion.

These commands repeat existing samples. They do not provide new external
validation evidence or establish clinical fitness. No metrics are used to tune
the model. The report/checkpoint pins deliberately fail when artifacts change.

## Hosts and services

| Host | SSH | Repository | Required service |
| --- | --- | --- | --- |
| `.1` proxy | `software@100.64.0.1` | `/home/software/Documents/.aarav/brain` | `brain-mri-tailnet-proxy` |
| `.7` app | `software@100.64.0.7` | `/home/software/Documents/.aa/brain` | `brain-mri-prototype`, `brain-mri-ollama` |
| `.3` excluded | `theaa@100.64.0.3` | `/home/theaa/Documents/brain-mri-data` | None for this prototype |

The app and proxy are user systemd services. They are enabled and user lingering
is enabled on `.1` and `.7`, allowing startup without interactive login. No reboot
was performed in the finishing pass; enabled state is not a completed reboot test.

The historical System status collector still reads `.1` and `.3` telemetry.
Its worker count is not the app readiness indicator: `.3` can be unavailable
while inference on `.7` remains ready. Collection does not launch training.

```bash
ssh software@100.64.0.7 'systemctl --user status brain-mri-prototype brain-mri-ollama --no-pager'
ssh software@100.64.0.7 'journalctl --user -u brain-mri-prototype -n 60 --no-pager'
ssh software@100.64.0.1 'systemctl --user status brain-mri-tailnet-proxy --no-pager'
```

If recovery is necessary, wait for active inference to finish, then restart the
affected service with `systemctl --user restart SERVICE`. A brief proxy 502 while
the application starts is expected; wait for `/api/capabilities` to report ready
before testing. Completed jobs are restored on startup until expiry; incomplete
jobs are removed. Do not restart merely because the GPU is idle.

## Deploy and roll back

Review `git status` before deployment: this repository contains unrelated local
work that must not be included accidentally. Deploy only reviewed files to the
correct host. Keep the current checkpoint in place; its SHA-256 is pinned:

`121422a861bbe7affaa5e161058e69eea737b2390651c3c03ea20256969e99e5`

Before replacing a deployed file, compare it with the previous committed version
to avoid overwriting remote edits. Preserve the previous version for rollback.
Restart the app only when idle, then run both acceptance commands. If they fail
because of the change, restore the previous deployed file and restart. Do not
overwrite the entire remote repository or reset its worktree.

## Completion and future work

Acceptance means the regression suite passes, both live sample workflows pass,
downloads match receipts, LLM rendering validates, temporary results are cleared,
and the browser shows the usable workflow. It does not mean every possible MRI
will be segmented correctly: the external cohort contained six weak cases,
including one reproducible small-lesion miss.

Further independent case testing and expert review remain useful. Retraining is
an optional future project, requiring a development-only experiment and a new
untouched cohort. Keep all training off unless a new experiment is explicitly
authorized. Never restart `.3` training as an operational recovery step.

The [independent-validation readiness record and review worksheet](independent-validation-review.md)
document the current data limitation and the inputs needed for the next evaluation.
