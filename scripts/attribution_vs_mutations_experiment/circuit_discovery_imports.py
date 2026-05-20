"""Re-export circuit_discovery_experiment modules used by this experiment."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CIRCUIT_DISCOVERY_DIR = _REPO_ROOT / "scripts" / "circuit_discovery_experiment"


def _ensure_circuit_discovery_on_path() -> None:
    for path in (_REPO_ROOT, _CIRCUIT_DISCOVERY_DIR):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


_ensure_circuit_discovery_on_path()

from attribution_patching_utils import create_induction_dataset_pandas  # noqa: E402
from EAP_dataset import EAPDataset  # noqa: E402

__all__ = ["EAPDataset", "create_induction_dataset_pandas"]
