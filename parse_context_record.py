#!/usr/bin/env python3
"""
parse_context_record.py
───────────────────────
Parses a completed Supernote .note file (or rasterised PNG) of the
archaeological Context Record form. Rasterises the note to a PIL Image,
crops each form field's region of interest, runs OCR on each crop, and
outputs structured data as JSON and/or CSV.

Supported OCR engines  (select with --engine)
---------------------------------------------
  ocrmac     Apple Vision framework. macOS only. Best default on Mac.
             pip install ocrmac

  rapidocr   RapidOCR via ONNX Runtime. Cross-platform (Linux/Windows/Mac).
             pip install rapidocr onnxruntime

  qwen       Qwen2.5-VL vision-language model, fully local.
             Best handwriting accuracy; ~8 GB RAM for 3B, ~16 GB for 7B.
             pip install git+https://github.com/huggingface/transformers
             pip install torch accelerate qwen-vl-utils

Platform guidance
-----------------
  macOS (Apple Silicon)  →  --engine ocrmac   (auto-detected)
                             --engine qwen     (better accuracy, slower)
  Linux / Windows        →  --engine rapidocr  (auto-detected)
                             --engine qwen     (better accuracy, needs RAM)

Usage
-----
  python parse_context_record.py context_001.note
  python parse_context_record.py context_001.note --engine qwen
  python parse_context_record.py context_001.note --engine qwen --qwen-model 7B
  python parse_context_record.py context_001.note --engine rapidocr
  python parse_context_record.py --batch ./field_notes/ --engine qwen
  python parse_context_record.py context_001.note --format json
  python parse_context_record.py context_001.note --debug-crops

Output
------
  context_001.json             — one JSON object per form
  context_records_batch.csv    — appended CSV row (shared across batch runs)

Requires context_record_schema.py in the same directory.
"""

import argparse
import csv
import io
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

# ── Shared schema ─────────────────────────────────────────────────────────────
try:
    from context_record_schema import (
        CANVAS, FIELD_REGIONS, OUTPUT_FIELDS, CHECKBOX_HINTS, NUMERIC_FIELDS,
        CSV_METADATA_KEYS, CSV_METADATA_LABELS, CSV_FIELD_KEYS, CSV_FIELD_LABELS,
    )
except ImportError:
    print("ERROR: context_record_schema.py not found. Place it alongside this script.")
    sys.exit(1)

# ── Available engine names ────────────────────────────────────────────────────
ENGINES = ["ocrmac", "rapidocr", "qwen"]

# ── Qwen model size → HuggingFace repo ───────────────────────────────────────
QWEN_MODELS = {
    "3B":  "Qwen/Qwen2.5-VL-3B-Instruct",
    "7B":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",
}

# ── numpy import (required by rapidocr) ──────────────────────────────────────
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Image loading and cropping
# ─────────────────────────────────────────────────────────────────────────────

def rasterise_note(note_path: Path, device: str) -> Image.Image:
    """Convert a .note file to a PIL Image via supernote-tool."""
    try:
        import supernotelib  # noqa: F401
    except ImportError:
        raise ImportError("Run:  pip install supernotelib")

    target = CANVAS[device]

    with tempfile.TemporaryDirectory() as tmp:
        out_png = Path(tmp) / "page.png"
        last_result = None

        for cmd in (
            ["supernote-tool", "convert", str(note_path), str(out_png)],
            [sys.executable, "-m", "supernotelib.cli", "convert",
             str(note_path), str(out_png)],
        ):
            try:
                r = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                last_result = r
                if r.returncode == 0 and out_png.exists():
                    break
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    "supernote-tool timed out after 60 s. "
                    "Check that the .note file is not corrupt."
                )
        else:
            stderr = last_result.stderr if last_result else "(command not found)"
            raise RuntimeError(
                f"supernote-tool failed.\nSTDERR: {stderr}\n"
                "Make sure supernotelib is installed:  pip install supernotelib"
            )

        img = Image.open(str(out_png)).convert("RGB")

    if img.size != target:
        print(f"WARNING: Rasterised image {img.size} differs from "
              f"{device} canvas {target}. Resizing — ROI alignment may be affected.")
        img = img.resize(target, Image.LANCZOS)
    return img


def load_image(path: Path, device: str) -> Image.Image:
    """Load a .note, .png, .jpg, etc. and return a PIL Image at canvas size."""
    target = CANVAS[device]
    if path.suffix.lower() == ".note":
        return rasterise_note(path, device)
    img = Image.open(str(path)).convert("RGB")
    if img.size != target:
        print(f"WARNING: Input image size {img.size} differs from "
              f"{device} canvas {target}. Resizing — ROI alignment may be affected. "
              f"Use --device to select the correct device model.")
        img = img.resize(target, Image.LANCZOS)
    return img


def crop_roi(img: Image.Image, roi: tuple, device: str) -> Image.Image:
    """Crop a fractional ROI from the full canvas image."""
    w, h = CANVAS[device]
    x0, y0, x1, y1 = roi
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


# ─────────────────────────────────────────────────────────────────────────────
# Engine: ocrmac  (macOS / Apple Vision)
# ─────────────────────────────────────────────────────────────────────────────

def _ocr_ocrmac(crop: Image.Image) -> str:
    """OCR via Apple Vision framework (macOS only)."""
    try:
        from ocrmac import ocrmac as _om
    except ImportError:
        raise ImportError(
            "ocrmac not installed.\nRun:  pip install ocrmac\n"
            "(macOS only — use --engine rapidocr or --engine qwen elsewhere)"
        )
    annotations = _om.OCR(
        crop,
        recognition_level="accurate",
        language_preference=["en-US"],
    ).recognize()
    return " ".join(
        t.strip() for t, conf, _ in annotations
        if t.strip() and conf > 0.3
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine: rapidocr  (cross-platform ONNX)
# ─────────────────────────────────────────────────────────────────────────────

_RAPIDOCR_ENGINE = None

def _ocr_rapidocr(crop: Image.Image) -> str:
    """OCR via RapidOCR + ONNX Runtime."""
    global _RAPIDOCR_ENGINE

    if not _NUMPY_AVAILABLE:
        raise ImportError(
            "numpy is required for rapidocr.\n"
            "Run:  pip install numpy rapidocr onnxruntime"
        )

    if _RAPIDOCR_ENGINE is None:
        try:
            from rapidocr import RapidOCR
        except ImportError:
            raise ImportError(
                "RapidOCR not installed.\n"
                "Run:  pip install rapidocr onnxruntime"
            )
        _RAPIDOCR_ENGINE = RapidOCR()

    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    arr = np.array(Image.open(buf).copy())

    result = _RAPIDOCR_ENGINE(arr)
    if result is None or result.txts is None:
        return ""
    return " ".join(
        t for t, score in zip(result.txts, result.scores)
        if t.strip() and score > 0.3
    )


# ─────────────────────────────────────────────────────────────────────────────
# Engine: qwen  (Qwen2.5-VL, fully local)
# ─────────────────────────────────────────────────────────────────────────────

_QWEN_MODEL     = None
_QWEN_PROCESSOR = None
_QWEN_DEVICE    = None

OCR_PROMPT = (
    "This is a cropped region of a handwritten archaeological context record form. "
    "Transcribe every handwritten word exactly as written. "
    "Return only the transcribed text — no commentary, no labels, no formatting."
)


def _ensure_qwen(model_size: str = "3B") -> None:
    """Download (first run only) and load Qwen2.5-VL into memory."""
    global _QWEN_MODEL, _QWEN_PROCESSOR, _QWEN_DEVICE

    if _QWEN_MODEL is not None:
        return

    repo = QWEN_MODELS.get(model_size)
    if repo is None:
        raise ValueError(
            f"Unknown Qwen model size '{model_size}'. "
            f"Choose from: {list(QWEN_MODELS)}"
        )

    try:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        raise ImportError(
            "Qwen dependencies not installed. Run:\n"
            "  pip install git+https://github.com/huggingface/transformers\n"
            "  pip install torch accelerate qwen-vl-utils"
        )

    if torch.cuda.is_available():
        _QWEN_DEVICE = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _QWEN_DEVICE = "mps"
    else:
        _QWEN_DEVICE = "cpu"

    dtype = torch.bfloat16 if _QWEN_DEVICE != "cpu" else torch.float32
    print(f"[qwen] Loading {repo} on {_QWEN_DEVICE} ({dtype}) ...")
    print("[qwen] First run downloads the model to ~/.cache/huggingface/")

    _QWEN_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        repo, torch_dtype=dtype, device_map="auto",
    )
    _QWEN_PROCESSOR = AutoProcessor.from_pretrained(repo)
    print("[qwen] Model ready.")


def _ocr_qwen(crop: Image.Image, model_size: str = "3B") -> str:
    """OCR via Qwen2.5-VL running fully locally."""
    import torch
    from qwen_vl_utils import process_vision_info

    _ensure_qwen(model_size)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text",  "text": OCR_PROMPT},
    ]}]

    text_input = _QWEN_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _QWEN_PROCESSOR(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_QWEN_DEVICE)

    with torch.inference_mode():
        generated_ids = _QWEN_MODEL.generate(
            **inputs, max_new_tokens=256, do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    decoded = _QWEN_PROCESSOR.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0].strip() if decoded else ""


# ─────────────────────────────────────────────────────────────────────────────
# Unified OCR dispatch
# ─────────────────────────────────────────────────────────────────────────────

def ocr_crop(crop: Image.Image, engine: str, qwen_model: str = "3B") -> str:
    """Route a crop to the selected OCR engine and return recognised text."""
    if engine == "ocrmac":
        return _ocr_ocrmac(crop)
    elif engine == "rapidocr":
        return _ocr_rapidocr(crop)
    elif engine == "qwen":
        return _ocr_qwen(crop, model_size=qwen_model)
    else:
        raise ValueError(f"Unknown engine '{engine}'. Choose from: {ENGINES}")


# ─────────────────────────────────────────────────────────────────────────────
# Checkbox detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_checkboxes_by_column(
    crop: Image.Image,
    options: list,
    engine: str,
    qwen_model: str = "3B",
) -> list:
    """
    Identify ticked checkboxes by dividing the crop into per-option column slices.

    For ocrmac / rapidocr:
        The checkbox row is divided into N equal vertical slices, one per option.
        OCR runs on each slice independently so tick characters can only be
        attributed to their own option column — eliminating the proximity-window
        ambiguity of the previous heuristic approach.

    For qwen:
        The full crop is sent with a structured prompt listing the options.
        The model is instructed to express uncertainty by omitting ambiguous marks,
        reducing false positives at the cost of occasional false negatives (the
        correct trade-off for an archival record).
    """
    if engine == "qwen":
        return _detect_checkboxes_qwen(crop, options, qwen_model)
    else:
        return _detect_checkboxes_by_column_ocr(crop, options, engine, qwen_model)


def _detect_checkboxes_by_column_ocr(
    crop: Image.Image,
    options: list,
    engine: str,
    qwen_model: str,
) -> list:
    """Column-slice checkbox detection for ocrmac and rapidocr."""
    w, h   = crop.size
    n      = len(options)
    ticked = []
    tick_chars = {"✓", "✗", "x", "☑", "■", "✔", "v", "[x]", "[v]"}

    for i, opt in enumerate(options):
        x0 = int(i * w / n)
        x1 = int((i + 1) * w / n)
        slice_crop = crop.crop((x0, 0, x1, h))
        raw = ocr_crop(slice_crop, engine, qwen_model).lower()
        if any(t in raw for t in tick_chars):
            ticked.append(opt)

    return ticked


def _detect_checkboxes_qwen(
    crop: Image.Image,
    options: list,
    qwen_model: str,
) -> list:
    """Qwen-based checkbox detection via structured prompt."""
    import torch
    from qwen_vl_utils import process_vision_info

    _ensure_qwen(qwen_model)

    opts_str = ", ".join(options)
    prompt = (
        f"This image shows a row of checkboxes from an archaeological form. "
        f"The options in left-to-right order are: {opts_str}. "
        f"List only the options that have a clear tick, cross, or mark inside their box. "
        f"If you are not certain whether a mark is intentional, do not include that option. "
        f"Return a comma-separated list, or the single word NONE if nothing is ticked."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop},
        {"type": "text",  "text": prompt},
    ]}]

    text_input = _QWEN_PROCESSOR.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _QWEN_PROCESSOR(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(_QWEN_DEVICE)

    with torch.inference_mode():
        generated_ids = _QWEN_MODEL.generate(
            **inputs, max_new_tokens=64, do_sample=False,
        )

    trimmed  = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    response = _QWEN_PROCESSOR.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    if response.upper() == "NONE" or not response:
        return []
    response_lower = response.lower()
    return [opt for opt in options if opt.lower() in response_lower]


# ─────────────────────────────────────────────────────────────────────────────
# Above-note extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_above_note(raw_text: str) -> str:
    """
    Extract the context number from an 'above_note_roi' OCR result.

    Recorders write a context number in the sub-row beneath 'Above' and annotate
    it 'This context', indicating stratigraphic self-equality (a standard Harris
    Matrix convention). Strip template noise and return the digit sequence only.
    """
    if not raw_text.strip():
        return ""
    digits = re.sub(r"\D", "", raw_text)
    return digits if digits else raw_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Numeric field confidence flagging
# ─────────────────────────────────────────────────────────────────────────────

def flag_numeric_uncertainty(record: dict) -> dict:
    """
    Flag numeric fields whose OCR result looks suspicious.

    Archaeological context numbers are 4-digit integers. Any value that contains
    non-digit characters (or is unexpectedly short) is marked with a low-confidence
    companion key so downstream reviewers can check against the physical record.

    Common OCR digit confusions: 8/9, 5/6, 0/O, 1/l.
    """
    for field in NUMERIC_FIELDS:
        val = record.get(field, "").strip()
        if val and not re.fullmatch(r"\d{4}", val):
            record[f"_{field}_confidence"] = "low — verify against physical record"
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Main parsing routine
# ─────────────────────────────────────────────────────────────────────────────

def parse_record(
    note_path: Path,
    device: str = "nomad",
    engine: str = "ocrmac",
    qwen_model: str = "3B",
    debug_crops: bool = False,
) -> dict:
    """Parse one completed context record form and return a structured dict."""
    print(f"[parse] Loading '{note_path.name}' ...")
    img = load_image(note_path, device)
    print(f"[parse] Canvas: {img.size}  |  engine: {engine}")

    if engine == "qwen":
        print(f"[parse] Qwen model: {QWEN_MODELS[qwen_model]}")
        _ensure_qwen(qwen_model)

    debug_dir = None
    if debug_crops:
        debug_dir = note_path.parent / f"{note_path.stem}_debug_crops"
        debug_dir.mkdir(exist_ok=True)
        print(f"[parse] Debug crops → {debug_dir}/")

    record = {
        "_source_file": str(note_path.name),
        "_device":      device,
        "_engine":      engine,
    }

    above_note_raw = ""

    for field_key, meta in FIELD_REGIONS.items():
        roi   = meta["roi"]
        ftype = meta["type"]
        crop  = crop_roi(img, roi, device)

        if debug_dir is not None:
            crop.save(str(debug_dir / f"{field_key}.png"))

        print(f"  [{field_key}] ...", end=" ", flush=True)

        if ftype == "above_note_roi":
            # Not a top-level output field — capture raw text for later processing
            above_note_raw = ocr_crop(crop, engine, qwen_model)
            print(f"→ (above_note raw: {above_note_raw!r})")
            continue

        elif ftype == "text":
            value = ocr_crop(crop, engine, qwen_model)

        elif ftype == "checkbox_group":
            value = detect_checkboxes_by_column(
                crop, meta.get("options", []), engine, qwen_model
            )

        else:
            value = ocr_crop(crop, engine, qwen_model)

        record[field_key] = value
        print(f"→ {str(value)[:80].replace(chr(10), ' ')!r}")

    # Attach above_note as a properly scoped field (not a peer of above/below)
    above_note = extract_above_note(above_note_raw)
    if above_note:
        record["above_note"] = above_note
        print(f"  [above_note] → {above_note!r}  (stratigraphic self-equality annotation)")

    # Flag suspicious numeric fields
    record = flag_numeric_uncertainty(record)

    return record


# ─────────────────────────────────────────────────────────────────────────────
# Output serialisers
# ─────────────────────────────────────────────────────────────────────────────

def save_json(record: dict, out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[output] JSON → {out_path}")


def save_csv(record: dict, out_path: Path) -> None:
    """
    Append one record row to the batch CSV.

    Detects schema mismatches between the current FIELD_REGIONS and an
    existing CSV file (e.g. if a field was added mid-project), and warns
    the user rather than silently producing misaligned columns.
    """
    all_keys   = CSV_METADATA_KEYS   + CSV_FIELD_KEYS
    all_labels = CSV_METADATA_LABELS + CSV_FIELD_LABELS

    file_exists = out_path.exists()

    if file_exists:
        with open(out_path, "r", encoding="utf-8") as f:
            reader       = csv.reader(f)
            existing_hdr = next(reader, [])
        if existing_hdr and existing_hdr != all_labels:
            print(
                f"WARNING: CSV schema mismatch in {out_path}.\n"
                f"  Existing : {existing_hdr}\n"
                f"  Current  : {all_labels}\n"
                f"  Use --outdir to write a fresh file, or manually delete the CSV."
            )

    row = []
    for k in all_keys:
        v = record.get(k, "")
        if isinstance(v, list):
            v = "; ".join(v)
        elif isinstance(v, dict):
            v = json.dumps(v)
        row.append(v)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(all_labels)
        writer.writerow(row)

    print(f"[output] CSV  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Engine auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def _default_engine() -> str:
    """Pick a sensible default engine based on platform and installed packages."""
    if platform.system() == "Darwin":
        try:
            from ocrmac import ocrmac  # noqa: F401
            return "ocrmac"
        except ImportError:
            pass
    # Non-Mac, or ocrmac unavailable on Mac
    try:
        from rapidocr import RapidOCR  # noqa: F401
        return "rapidocr"
    except ImportError:
        pass
    # Last resort: qwen will produce a helpful ImportError at first call
    return "qwen"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse completed Supernote context-record forms to JSON/CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Engine install commands
-----------------------
  ocrmac    (macOS only)       pip install ocrmac
  rapidocr  (cross-platform)   pip install rapidocr onnxruntime
  qwen      (all platforms, best accuracy, needs RAM)
            pip install git+https://github.com/huggingface/transformers
            pip install torch accelerate qwen-vl-utils

Qwen model sizes
----------------
  3B  — ~6 GB download,  ~8 GB RAM   (default; good for laptops)
  7B  — ~14 GB download, ~16 GB RAM  (better accuracy)
  72B — research/server use only
""",
    )
    parser.add_argument(
        "input",
        help="Path to a .note file, a rasterised PNG, or a folder (with --batch).",
    )
    parser.add_argument(
        "--engine", choices=ENGINES, default=None,
        help="OCR engine. Auto-detected if omitted (ocrmac on Mac, rapidocr elsewhere).",
    )
    parser.add_argument(
        "--qwen-model", choices=list(QWEN_MODELS), default="3B", metavar="SIZE",
        help="Qwen model size (3B / 7B / 72B). Only used with --engine qwen. Default: 3B.",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Process all .note files found in the input folder.",
    )
    parser.add_argument(
        "--device", choices=list(CANVAS), default="nomad",
        help="Supernote device model (affects canvas dimensions). Default: nomad.",
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "both"], default="both",
        help="Output format. Default: both.",
    )
    parser.add_argument(
        "--outdir", default=None,
        help="Directory to write output files (default: same as input).",
    )
    parser.add_argument(
        "--debug-crops", action="store_true",
        help="Save each field's cropped image for ROI debugging.",
    )
    args = parser.parse_args()

    engine     = args.engine or _default_engine()
    qwen_model = args.qwen_model
    input_path = Path(args.input)
    outdir     = Path(args.outdir) if args.outdir else None

    print(f"Engine: {engine}"
          + (f" ({QWEN_MODELS[qwen_model]})" if engine == "qwen" else ""))

    def run_one(note_path: Path, batch_csv: Path) -> None:
        record = parse_record(
            note_path,
            device=args.device,
            engine=engine,
            qwen_model=qwen_model,
            debug_crops=args.debug_crops,
        )
        base_dir = outdir or note_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)
        if args.format in ("json", "both"):
            save_json(record, base_dir / f"{note_path.stem}.json")
        if args.format in ("csv", "both"):
            save_csv(record, batch_csv)

    if args.batch:
        if not input_path.is_dir():
            print(f"ERROR: --batch requires a directory, got: {input_path}",
                  file=sys.stderr)
            sys.exit(1)
        note_files = sorted(input_path.glob("*.note"))
        if not note_files:
            print(f"No .note files found in '{input_path}'.", file=sys.stderr)
            sys.exit(1)
        print(f"Batch mode: {len(note_files)} file(s).")
        batch_csv = (outdir or input_path) / "context_records_batch.csv"
        for nf in note_files:
            run_one(nf, batch_csv)
    else:
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        batch_csv = (outdir or input_path.parent) / "context_records_batch.csv"
        run_one(input_path, batch_csv)

    print("\nDone.")


if __name__ == "__main__":
    main()
