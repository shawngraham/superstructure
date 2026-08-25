#!/usr/bin/env python3
"""
parse_hybrid_record.py
──────────────────────
A robust Hybrid AI parser for Supernote RTR context records on Apple Silicon.

Uses `ocrmac` for spatial layout detection and `mlx-lm` (Apple MLX framework)
to align the OCR spatial hints with the highly accurate Supernote Real-Time
Recognition (RTR) pen-stroke text dump.

Designed to handle both standard instruct models (e.g. Llama 3) and
reasoning-trace models (e.g. Qwen3.5/DeepSeek-Flash) without modification.

Dependencies (macOS only):
    pip install supernotelib pillow ocrmac mlx-lm

Usage:
    python parse_hybrid_record.py context_001.note
    python parse_hybrid_record.py context_001.note --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
    python parse_hybrid_record.py context_001.note --model Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit
    python parse_hybrid_record.py context_001.note --max-tokens 6000
    python parse_hybrid_record.py context_001.note --debug-crops

Requires context_record_schema.py in the same directory.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from PIL import Image

# ── Shared schema ─────────────────────────────────────────────────────────────
try:
    from context_record_schema import (
        CANVAS, FIELD_REGIONS, OUTPUT_FIELDS, OUTPUT_SCHEMA,
        CHECKBOX_HINTS, NUMERIC_FIELDS,
    )
except ImportError:
    print("ERROR: context_record_schema.py not found. Place it alongside this script.")
    sys.exit(1)

# ── Dependency checks ─────────────────────────────────────────────────────────

try:
    import supernotelib as sn
    from supernotelib.converter import TextConverter, ImageConverter
except ImportError:
    print("ERROR: supernotelib missing.  Run: pip install supernotelib pillow")
    sys.exit(1)

try:
    from ocrmac import ocrmac
except ImportError:
    print("ERROR: ocrmac missing.  Run: pip install ocrmac  (macOS only)")
    sys.exit(1)

try:
    from mlx_lm import load, generate
except ImportError:
    print("ERROR: mlx-lm missing.  Run: pip install mlx-lm")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

# Default device; overridable via --device
DEFAULT_DEVICE = "nomad"

# Model-agnostic stop tokens: Llama 3, Qwen, DeepSeek, Mistral, Phi families
STOP_TOKENS = [
    "<|eot_id|>", "<|end_of_text|>", "<|im_end|>",
    "<|endoftext|>", "</s>", "[/INST]",
]

# Reasoning-trace models consume ~1200 tokens in <think> before the answer
DEFAULT_MAX_TOKENS = 4096

# ocrmac confidence floor — permissive because this data is a spatial hint only
OCR_CONFIDENCE_THRESHOLD = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Extract RTR text and page image
# ─────────────────────────────────────────────────────────────────────────────

def extract_rtr_data(note_path: Path, device: str) -> tuple[str, Image.Image]:
    """
    Extract Real-Time Recognition text and the visual image from a .note file.

    Returns:
        raw_text : Full RTR text (all pages concatenated).
        img      : PIL Image of page 0, resized to the canonical canvas size.

    Exits on any notebook loading or conversion failure.
    """
    canvas_w, canvas_h = CANVAS[device]
    print(f"[supernotelib] Loading: {note_path.name}  (device={device}, "
          f"canvas={canvas_w}×{canvas_h})")

    try:
        if hasattr(sn, "load_notebook"):
            notebook = sn.load_notebook(str(note_path))
        elif hasattr(sn, "load"):
            notebook = sn.load(str(note_path))
        else:
            notebook = sn.notebook.load(str(note_path))
    except Exception as e:
        print(f"ERROR: Could not open note file. Details: {e}")
        sys.exit(1)

    # RTR text — collect from all pages
    try:
        txt_converter = TextConverter(notebook)
        total_pages   = notebook.get_total_pages()
        texts = []
        for i in range(total_pages):
            res = txt_converter.convert(i)
            if isinstance(res, bytes):
                res = res.decode("utf-8", errors="ignore")
            if res and isinstance(res, str):
                cleaned = res.strip()
                if cleaned:
                    texts.append(cleaned)
        raw_text = "\n\n".join(texts).strip()
    except Exception as e:
        print(f"ERROR: RTR text extraction failed. Details: {e}")
        sys.exit(1)

    # Image — page 0 only (context sheets are single-page)
    try:
        img_converter = ImageConverter(notebook)
        total_pages   = notebook.get_total_pages()
        if total_pages > 1:
            print(f"WARNING: Note has {total_pages} pages. "
                  f"Spatial mapping uses page 0 only.")
        img = img_converter.convert(0).convert("RGB")
        if img.size != (canvas_w, canvas_h):
            print(f"WARNING: Rasterised image size {img.size} differs from "
                  f"{device} canvas ({canvas_w}×{canvas_h}). Resizing — "
                  f"ROI alignment may be affected.")
            img = img.resize((canvas_w, canvas_h), Image.LANCZOS)
    except Exception as e:
        print(f"ERROR: Image extraction failed. Details: {e}")
        sys.exit(1)

    return raw_text, img


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Spatial OCR via Apple Vision (ocrmac)
# ─────────────────────────────────────────────────────────────────────────────

def get_rough_spatial_data(img: Image.Image, device: str,
                           debug_dir: Path | None) -> dict:
    """
    Crop each field region and run Apple Vision OCR on it.

    This data is intentionally impure — it will contain printed template text
    and is blind to checkmarks. It is used only as a spatial sequencing hint
    for the LLM, not as authoritative field values.

    If debug_dir is set, each crop is saved there as <field_key>.png.
    """
    canvas_w, canvas_h = CANVAS[device]
    print("[ocrmac] Running spatial OCR on field regions...")
    rough_data = {}

    for field_key, meta in FIELD_REGIONS.items():
        x0, y0, x1, y1 = meta["roi"]
        crop = img.crop((
            int(x0 * canvas_w), int(y0 * canvas_h),
            int(x1 * canvas_w), int(y1 * canvas_h),
        ))

        if debug_dir is not None:
            crop.save(str(debug_dir / f"{field_key}.png"))

        try:
            annotations = ocrmac.OCR(
                crop,
                recognition_level="accurate",
                language_preference=["en-US"],
            ).recognize()
            raw_text = " ".join([
                t for t, conf, _ in annotations
                if conf > OCR_CONFIDENCE_THRESHOLD
            ])
        except Exception as e:
            print(f"  WARNING: ocrmac failed for '{field_key}'. Details: {e}")
            raw_text = ""

        rough_data[field_key] = raw_text.strip()

    return rough_data


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Post-processing utilities
# ─────────────────────────────────────────────────────────────────────────────

def strip_reasoning_traces(text: str) -> str:
    """
    Remove <think>...</think> blocks emitted by reasoning-trace models
    (Qwen3.5, DeepSeek-Flash). These appear before the actual answer and
    contain JSON-like fragments that corrupt regex extraction.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Fallback: some models prefix with a plain-text reasoning header
    text = re.sub(r"^(Reasoning|Thinking|Chain of thought):.*?\n\n", "",
                  text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def truncate_at_stop_token(text: str) -> str:
    """Apply model-agnostic stop token truncation."""
    for token in STOP_TOKENS:
        if token in text:
            text = text.split(token)[0]
    return text.strip()


def extract_json_payload(text: str) -> str | None:
    """
    Extract a JSON object from model output using a layered strategy.

    Strategy 1: Find the LAST well-formed {...} block. Reasoning traces deposit
    their valid JSON at the end; taking the last block avoids fragments from
    inside the trace.

    Strategy 2: Strip markdown fences and extract outermost braces.

    Returns the raw JSON string, or None if nothing parseable was found.
    """
    # Strategy 1: last valid {...} block
    all_blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL)
    for candidate in reversed(all_blocks):
        if len(candidate) > 20:
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    # Strategy 2: strip fences, extract outermost braces
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        candidate = cleaned[start:end]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def validate_and_coerce_output(raw: dict) -> dict:
    """
    Validate and coerce model output against the schema.

    - Missing keys → empty string / empty list
    - Checkbox fields: accepts list, dict {opt: bool}, or comma string
    - Text fields: cast to str, strip whitespace
    - Hallucinated keys are discarded
    """
    validated = {}
    for key, meta in OUTPUT_FIELDS.items():
        value = raw.get(key)

        if meta["type"] == "checkbox_group":
            valid_options = meta["options"]
            if isinstance(value, list):
                coerced = [
                    opt for opt in valid_options
                    if any(str(v).strip().lower() == opt.lower() for v in value)
                ]
            elif isinstance(value, dict):
                coerced = [
                    opt for opt in valid_options
                    if value.get(opt) or value.get(opt.lower())
                ]
            elif isinstance(value, str) and value:
                coerced = [
                    opt for opt in valid_options
                    if opt.lower() in value.lower()
                ]
            else:
                coerced = []
            validated[key] = coerced

        else:
            validated[key] = "" if value is None else str(value).strip()

    return validated


def flag_numeric_uncertainty(validated: dict, rough_data: dict) -> dict:
    """
    Flag fields that should contain context numbers when RTR and OCR disagree.

    Archaeological context numbers are 4-digit integers. When the two sources
    produce different readings for the same field, the discrepancy is recorded
    in a companion _<field>_conflict key so downstream consumers can review it.

    Handles the common 8/9/5/6 digit confusion in both handwriting recognition
    and OCR.
    """
    for field in NUMERIC_FIELDS:
        rtr_val = validated.get(field, "").strip()
        ocr_val = rough_data.get(field, "").strip()

        # Extract digit sequences from each source for comparison
        rtr_digits = re.sub(r"\D", "", rtr_val)
        ocr_digits = re.sub(r"\D", "", ocr_val)

        # Flag if both sources found a number but they disagree
        if rtr_digits and ocr_digits and rtr_digits != ocr_digits:
            validated[f"_{field}_conflict"] = {
                "rtr": rtr_val,
                "ocr": ocr_val,
                "note": "Digit mismatch between RTR and OCR — verify against physical record.",
            }

    return validated


def extract_above_note(rough_data: dict) -> str:
    """
    Extract and interpret the 'This context' sub-annotation from the Above row.

    Recorders sometimes write a context number in the sub-row beneath 'Above'
    and annotate it 'This context', indicating stratigraphic self-equality.
    This is a standard Harris Matrix convention.

    Returns the context number as a string, or "" if the annotation is absent.
    """
    raw = rough_data.get("_above_note_roi", "").strip()
    if not raw:
        return ""
    # Strip any printed template text; extract a digit sequence
    digits = re.sub(r"\D", "", raw)
    return digits if digits else raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: LLM alignment via Apple MLX
# ─────────────────────────────────────────────────────────────────────────────

def refine_with_mlx(rough_data: dict, rtr_text: str,
                    model_id: str, max_tokens: int) -> dict:
    """
    Use an Apple MLX LLM to align RTR pen-stroke data with spatial field regions.

    Compatible with standard instruct models and reasoning-trace models.
    The prompt injects the exact key schema and checkbox option lists so the
    model cannot drift on key names, value types, or checkbox option names.
    Post-processing strips reasoning traces, applies model-agnostic stop token
    truncation, extracts JSON via layered fallback, and validates the schema.
    """
    print(f"[mlx-lm] Loading model: {model_id}")
    print(f"[mlx-lm] Token budget : {max_tokens} (includes any reasoning trace)")
    model, tokenizer = load(model_id)

    schema_description   = json.dumps(OUTPUT_SCHEMA,   indent=2)
    checkbox_description = json.dumps(CHECKBOX_HINTS,  indent=2)

    # Exclude internal ROI helper keys from the spatial map sent to the LLM
    llm_rough_data = {k: v for k, v in rough_data.items() if not k.startswith("_")}

    prompt = f"""You are a JSON data alignment API for archaeological field records.
You will be given two data sources extracted from a handwritten form and must return a single JSON object.

═══════════════════════════════════════
OUTPUT SCHEMA (mandatory — use EXACTLY these key names, no others):
═══════════════════════════════════════
{schema_description}

═══════════════════════════════════════
CHECKBOX FIELD OPTIONS (for checkbox_group fields, the ONLY valid values are):
═══════════════════════════════════════
{checkbox_description}
The options are listed in their LEFT-TO-RIGHT order on the physical form.
A solitary "X", "x", or "v" in the pen strokes corresponds to the option at
that horizontal position within the checkbox group.

═══════════════════════════════════════
DATA SOURCE 1 — ROUGH SPATIAL MAP (ocrmac)
═══════════════════════════════════════
Generated by image-scanning specific bounding boxes on the form.
IMPORTANT WARNINGS:
  - Contains the printed background template text as OCR noise.
  - Physically blind to handwritten checkmarks.
  - Use only as a spatial sequencing hint, not as ground-truth values.
{json.dumps(llm_rough_data, indent=2)}

═══════════════════════════════════════
DATA SOURCE 2 — PEN STROKES / RTR TEXT (ground truth)
═══════════════════════════════════════
Extracted directly from the stylus sensor. Contains ONLY what the user wrote.
Ignores all printed template text. Reading order is generally top-to-bottom,
but may reflect stroke-deposit order if the user filled fields out of sequence.
Checkmarks appear as solitary "X", "x", or "v" characters.
---
{rtr_text}
---

═══════════════════════════════════════
YOUR TASK
═══════════════════════════════════════
1. Align the PEN STROKES to the output schema keys using the ROUGH SPATIAL MAP
   as a sequencing reference only, not a value source.
2. Discard all OCR spatial map text. Replace entirely with pen stroke values.
3. For checkbox_group fields: identify solitary mark characters at the
   appropriate position in sequence. Map them to options using the positional
   order defined above. Output an array of matched option strings only.
4. For text fields: output a plain string. Empty string if nothing was written.
5. If the pen strokes contain a number followed by "This context" near the
   "above" field, ignore it — it is handled separately.
6. Return ONLY a raw JSON object. No explanation, no markdown, no preamble."""

    messages = [
        {"role": "system", "content": "You are a robotic JSON API. Output only raw JSON with no markdown fences, no preamble, no explanation."},
        {"role": "user",   "content": prompt},
    ]

    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as e:
        print(f"ERROR: Chat template application failed. Details: {e}")
        sys.exit(1)

    print("[mlx-lm] Generating alignment "
          "(reasoning trace will be stripped automatically)...")

    try:
        raw_response = generate(
            model, tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
    except Exception as e:
        print(f"ERROR: mlx-lm generation failed. Details: {e}")
        sys.exit(1)

    # Post-processing pipeline
    cleaned  = truncate_at_stop_token(raw_response)
    cleaned  = strip_reasoning_traces(cleaned)
    json_str = extract_json_payload(cleaned)

    if json_str is None:
        print("\nERROR: Could not extract a valid JSON object from model output.")
        print("─── Raw model output ─────────────────────────────────────────────")
        print(raw_response)
        print("──────────────────────────────────────────────────────────────────")
        print(f"  Suggestion: increase --max-tokens (current: {max_tokens})")
        sys.exit(1)

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"\nERROR: Extracted JSON string failed to parse. Details: {e}")
        print("─── Extracted string ─────────────────────────────────────────────")
        print(json_str)
        sys.exit(1)

    validated = validate_and_coerce_output(parsed)
    validated = flag_numeric_uncertainty(validated, rough_data)

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        print("\n=== Supernote Hybrid Parser (ocrmac + mlx-lm) ===")
        print("Usage: python parse_hybrid_record.py <note.note> [options]")
        print("\nOptions:")
        print("  --model MODEL_ID     HuggingFace model ID for mlx-lm")
        print("  --max-tokens N       Token budget (default: 4096)")
        print("  --device nomad|manta Supernote device model (default: nomad)")
        print("  --output PATH        Output JSON path")
        print("  --debug-crops        Save per-field crop images for ROI tuning")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Parse Supernote RTR archaeological context sheets (ocrmac + mlx-lm).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported model families:
  Standard instruct : mlx-community/Meta-Llama-3-8B-Instruct-4bit (default, ~5 GB)
  Reasoning-trace   : Jackrong/MLX-Qwen3.5-9B-DeepSeek-V4-Flash-4bit (~6 GB)

Reasoning-trace models require --max-tokens 4096+ to accommodate the
<think>...</think> preamble before the JSON answer.
        """,
    )
    parser.add_argument("input",          help="Path to a .note file.")
    parser.add_argument("--model",        default="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                        help="mlx-lm model ID. (default: Meta-Llama-3-8B-Instruct-4bit)")
    parser.add_argument("--max-tokens",   type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max generation tokens. (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--device",       choices=list(CANVAS), default=DEFAULT_DEVICE,
                        help="Supernote device model. (default: nomad)")
    parser.add_argument("--output",       default=None,
                        help="Output JSON path. Default: <input_stem>.json")
    parser.add_argument("--debug-crops",  action="store_true",
                        help="Save per-field crop images to <stem>_debug_crops/")
    args = parser.parse_args()

    note_path = Path(args.input)
    if not note_path.exists():
        print(f"ERROR: File not found: {note_path}")
        sys.exit(1)
    if note_path.suffix.lower() != ".note":
        print(f"WARNING: Expected a .note file, got '{note_path.suffix}'. Proceeding.")

    out_path  = Path(args.output) if args.output else note_path.parent / f"{note_path.stem}.json"

    debug_dir = None
    if args.debug_crops:
        debug_dir = note_path.parent / f"{note_path.stem}_debug_crops"
        debug_dir.mkdir(exist_ok=True)
        print(f"[debug] Crop images → {debug_dir}/")

    # Stage 1: RTR text + page image
    rtr_text, img = extract_rtr_data(note_path, args.device)

    print("\n─── RTR Pen Stroke Text ──────────────────────────────────────────")
    print(rtr_text if rtr_text else "(empty — enable Real-Time Recognition on device)")
    print("──────────────────────────────────────────────────────────────────\n")

    if not rtr_text:
        print("WARNING: No RTR text found. LLM alignment will be unreliable.")

    # Stage 2: Spatial OCR
    rough_data = get_rough_spatial_data(img, args.device, debug_dir)

    print("\n─── Rough Spatial Data (ocrmac) ──────────────────────────────────")
    print(json.dumps({k: v for k, v in rough_data.items() if not k.startswith("_")}, indent=2))
    print("──────────────────────────────────────────────────────────────────\n")

    # Stage 3: Extract above_note before sending rough_data to LLM
    above_note = extract_above_note(rough_data)

    # Stage 4: LLM alignment
    structured_data = refine_with_mlx(rough_data, rtr_text, args.model, args.max_tokens)

    # Stage 5: Attach above_note and provenance metadata
    if above_note:
        structured_data["above_note"] = above_note
    structured_data["_source_file"] = note_path.name
    structured_data["_device"]      = args.device
    structured_data["_model"]       = args.model
    structured_data["_engine"]      = "ocrmac + mlx-lm"

    # Stage 6: Write output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"\n[success] Structured record saved to: {out_path}")
    print("\n─── Final Output ─────────────────────────────────────────────────")
    print(json.dumps(structured_data, indent=2, ensure_ascii=False))
    print("──────────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
