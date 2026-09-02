import json
import os
import shutil
import subprocess
from pathlib import Path


def test_queue_keeps_each_seed_bound_to_its_output_directory(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / ".venv/bin").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "training").mkdir()
    (root / "test-bin").mkdir()
    shutil.copy(Path(__file__).parents[1] / "scripts/run_durable_cuda_queue.sh", root / "scripts")
    (root / "data/study.json").write_text("{}\n")
    (root / "training/profile.yaml").write_text("id: test\n")
    fake_python = root / ".venv/bin/python"
    fake_python.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "while (($#)); do case $1 in --seed) seed=$2; shift 2;; --output) output=$2; shift 2;; *) shift;; esac; done\n"
        "mkdir -p \"$output\"\n"
        "printf '{\"seed\": %s}\\n' \"$seed\" > \"$output/run.json\"\n"
        "printf '{}\\n' > \"$output/external.json\"\n"
    )
    fake_python.chmod(0o755)
    (root / "test-bin/flock").write_text("#!/usr/bin/env bash\nexit 0\n")
    (root / "test-bin/date").write_text("#!/usr/bin/env bash\nprintf '2026-09-01T00:00:00Z\\n'\n")
    (root / "test-bin/flock").chmod(0o755)
    (root / "test-bin/date").chmod(0o755)

    subprocess.run(
        [root / "scripts/run_durable_cuda_queue.sh", "identity-test", "data/study.json",
         "training/profile.yaml", "100", "brats:20260902", "brats:20260903"],
        check=True, cwd=root, env={**os.environ, "PATH": f"{root / 'test-bin'}:{os.environ['PATH']}"},
    )

    for seed in (20260902, 20260903):
        run = root / f"runs/identity-test--brats--{seed}--e100/run.json"
        assert json.loads(run.read_text())["seed"] == seed
    status = json.loads((root / "runs/queue-logs/identity-test.status.json").read_text())
    assert status["state"] == "complete"
    assert status["queuedRuns"] == []
    assert status["completedCount"] == status["totalCount"] == 2
