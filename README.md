# M-102 — Rad-ReStruct + OpenI Chest X-Ray Structured VQA

> **Note on naming**: the repo slug retains the original
> `candidptx_pneumothorax_rib_tube_seg` token so downstream identifiers
> stay stable, but the **pipeline content is structured chest-X-ray VQA
> built from Rad-ReStruct labels and the OpenI NLM CXR image pool**.
> CandidPTX (Auckland) raw bytes are not publicly auto-onrampable
> (PhysioNet-style credentialed access only), so M-102 ships radrestruct
> VQA samples under the same M-102 slot.

This repository is part of the Med-VR data-pipeline suite for the VBVR
(Very Big Video Reasoning) benchmark. It produces standardized video-
reasoning task samples from the underlying raw medical dataset.

## Task

**Prompt shown to the model**:

> This video shows a chest X-ray (Indiana University OpenI dataset) with a
> structured radiology report rendered as a sequence of question/answer
> panels (Rad-ReStruct schema). Each panel reveals one diagnostic question
> and its ground-truth answer in order: questions first appear alone
> (yellow), then the answer is revealed (green for positive findings,
> grey for negatives like "no" / "no selection"). For each question in
> the video, report the answer exactly as shown.

## S3 Raw Data

```
s3://med-vr-datasets/M-102/radrestruct_labels/   (cloned label repo)
s3://med-vr-datasets/M-102/openi_images/         (~7470 PNG)
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate samples (downloads raw from S3 on first run)
python examples/generate.py

# Generate only N samples
python examples/generate.py --num-samples 10

# Custom output directory
python examples/generate.py --output data/my_output
```

## Output Layout

```
data/questions/radrestruct_chest_xr_vqa_task/
├── radrestruct_<report_id>_<NNNNN>/
│   ├── first_frame.png
│   ├── final_frame.png
│   ├── first_video.mp4
│   ├── last_video.mp4
│   ├── ground_truth.mp4
│   ├── prompt.txt
│   └── metadata.json
├── ...
```

## Configuration

`src/pipeline/config.py` (`TaskConfig`):

| Field | Default | Description |
|---|---|---|
| `domain` | `"radrestruct_chest_xr_vqa"` | Task domain string used in output paths. |
| `s3_bucket` | `"med-vr-datasets"` | S3 bucket containing raw data. |
| `s3_prefix` | `"M-102/"` | S3 key prefix containing radrestruct_labels/ and openi_images/. |
| `fps` | `4` | Output video FPS. |
| `target_size` | `(640, 640)` | Output frame size. |
| `max_qa_pairs` | `6` | Max QA pairs rendered per sample. |
| `frames_per_panel` | `5` | Frames each panel is held. |

## Repository Structure

```
M-102_candidptx_pneumothorax_rib_tube_seg_data-pipeline/
├── core/                ← shared pipeline framework
├── eval/                ← shared evaluation utilities
├── src/
│   ├── download/
│   │   └── downloader.py   ← S3 raw-data downloader (uses aws s3 sync)
│   └── pipeline/
│       ├── config.py        ← task config
│       ├── pipeline.py      ← TaskPipeline
│       └── transforms.py    ← visualization helpers
├── examples/
│   └── generate.py
├── requirements.txt
├── README.md
└── LICENSE
```
