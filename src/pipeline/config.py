"""Pipeline configuration for M-102 Rad-ReStruct chest X-ray VQA."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Rad-ReStruct + OpenI structured chest X-ray VQA task config."""

    domain: str = Field(default="radrestruct_chest_xr_vqa")

    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(default="M-102/")  # contains radrestruct_labels/ + openi_images/
    fps: int = Field(default=4)
    raw_dir: Path = Field(default=Path("raw"))
    target_size: tuple = Field(default=(640, 640))
    # Max QA pairs to render per sample → keeps the GT video in the 5-15s window.
    max_qa_pairs: int = Field(default=6)
    # Frames each (question_only) and (question+answer) panel is held.
    frames_per_panel: int = Field(default=5)
