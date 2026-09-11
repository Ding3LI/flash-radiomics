from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def ensure_workspace_on_path() -> None:
    path_text = str(WORKSPACE_ROOT)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def relative_to_workspace(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def make_experiment_dir(tool_name: str, mode: str, backend: str) -> Path:
    exp_root = WORKSPACE_ROOT / "reproducibility" / tool_name / "exp"
    exp_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{mode}_{backend}"
    exp_dir = exp_root / base_name
    suffix = 1
    while exp_dir.exists():
        exp_dir = exp_root / f"{base_name}_{suffix:02d}"
        suffix += 1
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_feature_value_matrix(
    per_feature_rows: list[dict],
    value_key: str,
) -> tuple[list[str], list[dict]]:
    identity_fields = ["case_id", "label", "roi_name", "roi_voxels"]
    feature_keys = sorted({str(row["feature_key"]) for row in per_feature_rows})

    grouped: dict[tuple[str, int, str, int], dict] = {}
    for row in per_feature_rows:
        case_id = str(row["case_id"])
        label = int(row["label"])
        roi_name = str(row["roi_name"])
        roi_voxels = int(row.get("roi_voxels", 0))
        group_key = (case_id, label, roi_name, roi_voxels)
        group_row = grouped.setdefault(
            group_key,
            {
                "case_id": case_id,
                "label": label,
                "roi_name": roi_name,
                "roi_voxels": roi_voxels,
            },
        )
        group_row[str(row["feature_key"])] = row.get(value_key, float("nan"))

    ordered_rows = [grouped[key] for key in sorted(grouped)]
    return identity_fields + feature_keys, ordered_rows


def sum_mean_and_std(rows: list[dict], mean_key: str, std_key: str) -> tuple[float, float]:
    total_mean = 0.0
    total_variance = 0.0
    for row in rows:
        mean_value = row.get(mean_key, float("nan"))
        std_value = row.get(std_key, float("nan"))
        if isinstance(mean_value, (int, float)) and math.isfinite(float(mean_value)):
            total_mean += float(mean_value)
        if isinstance(std_value, (int, float)) and math.isfinite(float(std_value)):
            total_variance += float(std_value) ** 2
    return float(total_mean), float(math.sqrt(total_variance))
