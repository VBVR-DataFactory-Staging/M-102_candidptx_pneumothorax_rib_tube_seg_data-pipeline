"""Downloader for M-102 — Rad-ReStruct labels + OpenI chest X-ray images.

Lives on S3 under:
    s3://med-vr-datasets/M-102/radrestruct_labels/...   (full git clone)
    s3://med-vr-datasets/M-102/openi_images/*.png       (~7470 PNG)

Yields one dict per (radrestruct report-id, OpenI image) pairing.
"""
from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional


REQUIRED_QA_DIRS = ["train_qa_pairs", "val_qa_pairs", "test_qa_pairs"]


def _aws_sync(bucket: str, prefix: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync",
        f"s3://{bucket}/{prefix}",
        str(dst),
        "--only-show-errors",
        "--no-progress",
    ]
    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_raw(config) -> tuple[Path, Path, Path]:
    """Make sure radrestruct_labels/ and openi_images/ are on local disk.

    Returns (radrestruct_root, qa_root, openi_root).
    """
    raw_dir = Path(config.raw_dir)
    rr_root = raw_dir / "radrestruct_labels"
    openi_root = raw_dir / "openi_images"
    qa_root = rr_root / "data" / "radrestruct"

    need_rr = not qa_root.exists() or not any(
        (qa_root / d).exists() for d in REQUIRED_QA_DIRS
    )
    if need_rr:
        _aws_sync(config.s3_bucket, f"{config.s3_prefix.rstrip('/')}/radrestruct_labels", rr_root)
    else:
        print(f"[download] radrestruct_labels already present at {rr_root}")

    need_openi = not openi_root.exists() or len(list(openi_root.glob("*.png"))) < 100
    if need_openi:
        _aws_sync(config.s3_bucket, f"{config.s3_prefix.rstrip('/')}/openi_images", openi_root)
    else:
        print(f"[download] openi_images already present at {openi_root}")

    return rr_root, qa_root, openi_root


def _load_id_to_img(qa_root: Path) -> dict:
    p = qa_root / "id_to_img_mapping_frontal_reports.json"
    return json.loads(p.read_text())


def _iter_qa_files(qa_root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (report_id_str, qa_json_path) across train+val+test splits."""
    for split in REQUIRED_QA_DIRS:
        d = qa_root / split
        if not d.exists():
            continue
        for jf in sorted(d.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            yield jf.stem, jf


class TaskDownloader:
    def __init__(self, config):
        self.config = config

    def iter_samples(self, limit: Optional[int] = None) -> Iterator[dict]:
        rr_root, qa_root, openi_root = _ensure_raw(self.config)
        id_to_img = _load_id_to_img(qa_root)
        # Build {imgname_stem: full_path} for fast lookup.
        png_index: dict[str, Path] = {p.stem: p for p in openi_root.glob("*.png")}
        print(f"[download] found {len(png_index)} OpenI PNGs, {len(id_to_img)} report ids")

        emitted = 0
        for report_id, qa_path in _iter_qa_files(qa_root):
            img_names = id_to_img.get(report_id, [])
            if not img_names:
                continue
            # Pick the first image whose stem we actually have.
            img_path: Optional[Path] = None
            for nm in img_names:
                if nm in png_index:
                    img_path = png_index[nm]
                    break
            if img_path is None:
                continue
            try:
                qa_data = json.loads(qa_path.read_text())
            except Exception:
                continue
            if not isinstance(qa_data, list) or not qa_data:
                continue
            yield {
                "report_id": report_id,
                "image_path": str(img_path),
                "image_name": img_path.stem,
                "qa": qa_data,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return

    # Older harness API expected by some pipelines.
    def download(self, limit: Optional[int] = None):
        yield from self.iter_samples(limit=limit)


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
