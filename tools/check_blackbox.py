#!/usr/bin/env python3
"""Recompute the packaged blackbox timing summary; standard library only.

This checks the sanitized 130-row export against the existing rounded analysis.
It does not rerun the private client/service or prove the historical cleanup.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

SCHEMA = "kvbridge-offload-study.blackbox-durations.v1"
METRICS = {
    "mac_total_ms": ("mac_total_ms",),
    "server_window_ms": ("server_window_ms",),
    "server_dispatch_ms": ("dispatch_ms",),
    "server_egress_ms": ("server_egress_ms",),
    "kernel_ms": ("native", "kernel_event_ms"),
    "py_backend_ms": ("native", "python_backend_ms"),
}
OP_KEYS = {
    "cycle", "mac_total_ms", "client_total_ms", "dispatch_ms", "new_version",
    "new_boundary", "native", "server_window_ms", "server_egress_ms",
    "transport_client_residual_ms",
}
NATIVE_KEYS = {
    "python_backend_ms", "python_stage_ms", "kernel_event_ms", "pipeline_event_ms",
    "host_stage_ms", "h2d_event_ms", "d2h_event_ms", "kv_stage_ms", "tail_payload_bytes",
}


def finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: finite numeric value required")
    return float(value)


def percentile95(values: list[float]) -> float:
    """Nearest-rank p95: sorted[ceil(.95*n)-1], index 118 for 125 rows."""
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def recompute(document: dict) -> dict:
    if set(document) != {"schema", "evidence_kind", "transformation", "analysis_policy", "result", "ops"}:
        raise ValueError("unexpected/missing public export fields")
    if document.get("schema") != SCHEMA:
        raise ValueError("unexpected public blackbox schema")
    rows = document["ops"]
    if len(rows) != 130 or [r["cycle"] for r in rows] != list(range(130)):
        raise ValueError("expected all 130 unique cycles in original order, 0..129")
    policy = document["analysis_policy"]
    if policy.get("steady_cycle_begin") != 5 or policy.get("steady_cycle_end_inclusive") != 129:
        raise ValueError("steady sample selection changed")
    if set(document["result"]) != {"status", "cycles", "history_appended", "ops", "server_log_fetched"}:
        raise ValueError("unexpected/missing public result fields")
    if document["result"]["cycles"] != 130 or document["result"]["ops"] != 130:
        raise ValueError("result count does not match rows")
    for row in rows:
        if set(row) != OP_KEYS or set(row["native"]) != NATIVE_KEYS:
            raise ValueError(f"cycle {row['cycle']}: unexpected/missing fields (identifiers are not public)")
        for key, value in row.items():
            if key != "native":
                number = finite_number(value, key)
                if key != "transport_client_residual_ms" and number < 0:
                    raise ValueError(f"cycle {row['cycle']}: negative {key}")
        for key, value in row["native"].items():
            if finite_number(value, "native." + key) < 0:
                raise ValueError(f"cycle {row['cycle']}: negative native.{key}")
        if row["server_window_ms"] < row["server_egress_ms"] or row["server_egress_ms"] < 0:
            raise ValueError(f"cycle {row['cycle']}: invalid server duration ordering")
        residual = row["mac_total_ms"] - row["server_window_ms"]
        if not math.isclose(residual, row["transport_client_residual_ms"], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"cycle {row['cycle']}: paired residual formula mismatch")
    steady = rows[5:]
    report = {"joined": len(rows), "steady": len(steady), "p50": {}, "p95": {}}
    for name, path in METRICS.items():
        values = []
        for row in steady:
            value = row
            for key in path: value = value[key]
            values.append(float(value))
        report["p50"][name] = round(statistics.median(values), 3)
        report["p95"][name] = round(percentile95(values), 3)
    residuals = [r["transport_client_residual_ms"] for r in steady]
    report["transport_residual_p50"] = round(statistics.median(residuals), 3)
    report["transport_residual_p95"] = round(percentile95(residuals), 3)
    return report


def compare(actual, expected, path="") -> list[str]:
    if isinstance(actual, dict) and isinstance(expected, dict):
        failures = []
        for key in sorted(set(actual) | set(expected)):
            p = f"{path}.{key}" if path else key
            if key not in actual or key not in expected:
                failures.append(f"{p}: missing or unexpected field")
            else:
                failures.extend(compare(actual[key], expected[key], p))
        return failures
    return [] if actual == expected else [f"{path}: recomputed={actual!r}, published={expected!r}"]


def validate(data_dir: Path) -> dict:
    public = json.loads((data_dir / "blackbox-session-05-ops.json").read_text(encoding="utf-8"))
    expected = json.loads((data_dir / "blackbox-session-05-analysis.json").read_text(encoding="utf-8"))
    actual = recompute(public)
    differences = compare(actual, expected)
    if differences:
        raise ValueError("blackbox summary mismatch:\n" + "\n".join(differences))
    return actual


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args(argv)
    try:
        actual = validate(args.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: 130 joined rows, 125 steady rows; all 14 rounded timing statistics match")
    print(json.dumps(actual, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
