"""Frame/video transforms for M-102 Rad-ReStruct chest X-ray VQA.

Builds an animation that sequentially reveals (Question → Answer) panels
overlaid on the chest X-ray. Output is a single ground-truth video plus a
"first" (image only) and "last" (final-question) clip.
"""
from __future__ import annotations
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np


# ── colours (BGR for OpenCV) ─────────────────────────────────────────────
BG_PANEL = (28, 28, 28)
FG_TEXT = (240, 240, 240)
QUESTION_COLOR = (255, 200, 80)   # warm yellow
ANSWER_POS_COLOR = (90, 220, 110)   # green = positive finding
ANSWER_NEG_COLOR = (200, 200, 200)  # grey = negative / no selection
BORDER_COLOR = (40, 40, 40)


def load_and_resize(path: str, size: Tuple[int, int]) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3 and img.shape[2] == 3:
        pass
    else:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _wrap(text: str, width: int) -> List[str]:
    if not text:
        return [""]
    return textwrap.wrap(text, width=width) or [text[:width]]


def _put_text_block(
    canvas: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.62,
    thickness: int = 1,
    line_height: int = 22,
):
    x, y = origin
    for i, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _is_negative_answer(answer_list: List[str]) -> bool:
    if not answer_list:
        return True
    joined = " ".join(a.lower() for a in answer_list)
    return joined in {"no", "no selection", "none", ""}


def render_panel(
    base_img: np.ndarray,
    question: str,
    answer: Optional[str],
    qa_idx: int,
    qa_total: int,
) -> np.ndarray:
    """Compose one frame: chest X-ray on the left, Q&A panel on the right."""
    H, W = base_img.shape[:2]
    panel_w = max(420, W // 2)
    canvas_w = W + panel_w
    canvas = np.full((H, canvas_w, 3), BG_PANEL, dtype=np.uint8)
    canvas[:, :W] = base_img
    cv2.rectangle(canvas, (W, 0), (canvas_w - 1, H - 1), BORDER_COLOR, 2)

    pad = 20
    cur_y = pad + 20

    header = f"Q {qa_idx + 1}/{qa_total}"
    cv2.putText(
        canvas, header, (W + pad, cur_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, QUESTION_COLOR, 2, cv2.LINE_AA,
    )
    cur_y += 32

    q_lines = _wrap(question, width=max(28, panel_w // 14))
    _put_text_block(canvas, q_lines, (W + pad, cur_y), FG_TEXT, scale=0.62)
    cur_y += len(q_lines) * 22 + 20

    if answer is None:
        cv2.putText(
            canvas, "...", (W + pad, cur_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ANSWER_NEG_COLOR, 2, cv2.LINE_AA,
        )
    else:
        ans_color = ANSWER_NEG_COLOR if answer.strip().lower() in {"no", "no selection", "none", ""} else ANSWER_POS_COLOR
        cv2.putText(
            canvas, "Answer:", (W + pad, cur_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, FG_TEXT, 1, cv2.LINE_AA,
        )
        cur_y += 26
        a_lines = _wrap(answer, width=max(28, panel_w // 14))
        _put_text_block(canvas, a_lines, (W + pad, cur_y), ans_color, scale=0.7, thickness=2, line_height=26)

    return canvas


def select_qa_pairs(qa: list, max_pairs: int) -> List[dict]:
    """Pick the most informative QA tuples (prefer positive findings)."""
    norm: List[dict] = []
    for entry in qa:
        if not isinstance(entry, list) or len(entry) < 4:
            continue
        question, answer_list, _children, meta = entry[0], entry[1] or [], entry[2], entry[3] or {}
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer_list, list):
            continue
        norm.append({
            "question": question.strip(),
            "answer": ", ".join(str(a) for a in answer_list) if answer_list else "no",
            "negative": _is_negative_answer(answer_list),
            "meta": meta if isinstance(meta, dict) else {},
        })

    if not norm:
        return []

    positives = [q for q in norm if not q["negative"]]
    negatives = [q for q in norm if q["negative"]]
    chosen = positives[:max_pairs]
    if len(chosen) < max_pairs:
        chosen += negatives[: max_pairs - len(chosen)]
    return chosen[:max_pairs]


def build_frames(
    base_img: np.ndarray,
    qa_pairs: List[dict],
    frames_per_panel: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[np.ndarray]]:
    """Return (all_frames, first_frame_rgb, last_frame_rgb, last_segment_frames)."""
    frames: List[np.ndarray] = []
    last_segment: List[np.ndarray] = []
    n = len(qa_pairs)

    # Lead-in: image only with a "Begin" prompt.
    intro = render_panel(base_img, "Reviewing chest X-ray...", None, 0, max(1, n))
    for _ in range(frames_per_panel):
        frames.append(intro)

    for i, qa in enumerate(qa_pairs):
        q_only = render_panel(base_img, qa["question"], None, i, n)
        q_and_a = render_panel(base_img, qa["question"], qa["answer"], i, n)
        for _ in range(frames_per_panel):
            frames.append(q_only)
        for _ in range(frames_per_panel):
            frames.append(q_and_a)
        if i == n - 1:
            last_segment = [q_only] * frames_per_panel + [q_and_a] * frames_per_panel

    # ensure non-empty
    if not frames:
        frames = [intro]
    if not last_segment:
        last_segment = [frames[-1]]

    first_rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    last_rgb = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB)
    return frames, first_rgb, last_rgb, last_segment


def make_video(frames: Iterable[np.ndarray], out_path: Path, fps: int) -> None:
    frames = list(frames)
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait(timeout=180)
