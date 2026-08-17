#!/usr/bin/env python3
"""Run a fail-closed, non-promotable BraTS-only ROCm replication."""

from __future__ import annotations

if __package__:
    from .train_glioma import main
else:
    from train_glioma import main


if __name__ == "__main__":
    main(exploratory_rocm=True)
