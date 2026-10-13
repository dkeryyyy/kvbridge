#!/usr/bin/env python3
"""Offline public-package integrity, link, JSON and historical-data checks.

Run normally to validate the frozen package. Maintainers must explicitly pass
--write-manifest after reviewed edits to refresh hashes; a normal check never
rewrites files. No network, CUDA, model or parent repository is required.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE-MANIFEST.json"


def package_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(p in {".git", "__pycache__", ".DS_Store"} or p.startswith(("build", "cmake-build-")) for p in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in release: {rel}")
        if path.is_file() and path != MANIFEST:
            yield path


def entries():
    return [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in package_files()]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value {value} in {path.name}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject(value)
        return number
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject, parse_float=finite_float)


def validate_links():
    count = 0
    for path in package_files():
        if path.suffix != ".md":
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = target.strip().split(' "', 1)[0].strip("<>")
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            dest = (path.parent / target).resolve()
            require(dest.is_relative_to(ROOT), f"link escapes release: {path.name}: {target}")
            require(dest.exists(), f"broken link: {path.name}: {target}")
            count += 1
    return count


def validate_history():
    data = read_json(ROOT / "data/att-bench-65536-v2-07.json")
    require(not data["failure"], "historical benchmark recorded a failure")
    for name, variant in data["variants"].items():
        for direction in ("forward", "reverse"):
            samples = sorted(variant[direction + "_ms"])
            stats = variant["stats_" + direction]
            require(len(samples) == stats["samples"] == 32, f"sample count: {name}")
            require(all(math.isfinite(x) and x > 0 for x in samples), f"invalid timings: {name}")
            # Historical code uses nearest-rank percentiles, unlike the new
            # example's average-of-two median for even sample counts.
            require(math.isclose(samples[math.ceil(len(samples) * .5) - 1], stats["p50"], abs_tol=1e-9), f"median: {name}")
            require(math.isclose(samples[math.ceil(len(samples) * .95) - 1], stats["p95"], abs_tol=1e-9), f"p95: {name}")
            require(math.isclose(sum(samples)/len(samples), stats["mean"], abs_tol=1e-9), f"mean: {name}")
    a = data["variants"]["A_current_shared"]["stats_forward"]["p50"]
    b = data["variants"]["B4_mma_tensorcore"]["stats_forward"]["p50"]
    table = read_json(ROOT / "data/kv-memory-table.json")
    with (ROOT / "data/kv-memory-table.csv").open(newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == len(table["rows"]), "memory table row count")
    for row, csv_row in zip(table["rows"], csv_rows):
        size = row["tokens"] * 2080
        require(size == row["per_layer_bytes"], "KV memory byte formula")
        require(size / 2**20 == row["per_layer_MiB"], "KV memory MiB formula")
        require(int(csv_row["tokens"]) == row["tokens"], "CSV token count")
        for layers in (1, 2, 3, 4, 5, 16):
            expected = round(size * layers / 2**30, 4)
            require(row[f"{layers}_layers_GiB"] == expected, "KV memory GiB formula")
            require(float(csv_row[f"{layers}_layer{'s' if layers != 1 else ''}_GiB"]) == expected, "CSV memory formula")
    print(f"Historical 64Ki forward-median speedup recomputed: {a/b:.6f}x (old version only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    current = entries()
    if args.write_manifest:
        saved = {"schema": "kvbridge.release-manifest.v1", "version": "0.1.0-research",
            "scope": "All release files except this manifest; build/cache/git directories excluded.",
            "files": current, "total_files": len(current),
            "total_bytes": sum(e["bytes"] for e in current)}
    else:
        saved = read_json(MANIFEST)
    require(saved["files"] == current, "manifest stale or file changed; review edits before --write-manifest")
    require(saved["total_files"] == len(current), "manifest file count")
    require(saved["total_bytes"] == sum(e["bytes"] for e in current), "manifest byte total")
    json_count = 0
    for path in package_files():
        if path.suffix == ".json":
            read_json(path); json_count += 1
        require(path.suffix.lower() not in {".exe", ".dll", ".dylib", ".so", ".a", ".lib", ".o", ".obj", ".safetensors", ".gguf"},
                f"binary/model artifact in source release: {path.name}")
        require(path.stat().st_size < 5 * 1024 * 1024, f"unexpected large file: {path.name}")
    record = read_json(ROOT / "validation/host-checks.json")
    for item in record["compiled_sources"]:
        source = (ROOT / item["path"]).resolve()
        require(source.is_relative_to(ROOT), "host record source outside package")
        require(hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"],
                f"host validation source changed: {item['path']}")
    links = validate_links()
    validate_history()
    blackbox = ROOT / "tools/check_blackbox.py"
    if blackbox.exists():
        subprocess.run([sys.executable, str(blackbox)], cwd=ROOT, check=True)
    if args.write_manifest:
        temporary = MANIFEST.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
        temporary.replace(MANIFEST)
    print(f"PASS: {len(current)} manifest entries; {json_count} JSON files; {links} local Markdown links")
    print("Scope: offline package/data checks only; no CUDA compilation or GPU execution.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
