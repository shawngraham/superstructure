"""
context_record_schema.py
────────────────────────
Single source of truth for the archaeological context record form schema.

Both parse_hybrid_record.py and parse_context_record.py import from here.
Modify FIELD_REGIONS in this file only — changes propagate to both parsers.

ROI coordinates are fractional (0.0–1.0) of the canvas:
  Nomad : 1404 × 1872 px
  Manta : 1920 × 2560 px

Tune ROI values here; use --debug-crops on either script to inspect crops.

Key calibration notes (from test image Context_test_2, 2026-08-24):
  - site_code x0 pushed right to 0.185 to clear the dotted leader line.
  - All header-row fields have y1 expanded to 0.107 / 0.134 for Apple Vision.
  - feature_type x0 pulled left to 0.206 to capture the Deposit checkbox column.
  - this_context is a dedicated ROI for the "9555 This context" sub-annotation
    that recorders write inside the Above row to denote stratigraphic self-equality.
    It is stored as above_note in output, not as a peer field.
"""

# ── Canvas dimensions ────────────────────────────────────────────────────────

CANVAS = {
    "nomad": (1404, 1872),
    "manta": (1920, 2560),
}

# ── Field schema ─────────────────────────────────────────────────────────────
#
# type values:
#   "text"           — free-text handwriting; output is a string
#   "checkbox_group" — one or more ticked boxes; output is a list of strings
#   "above_note_roi" — special: captures the "X This context" sub-row inside
#                      the Above section; output stored as "above_note" string,
#                      not as an independent top-level field in the record

FIELD_REGIONS = {
    "site_code": {
        "label": "Site Code",
        # x0 pushed right to 0.185 to clear the dotted leader line artefact
        "roi":   (0.185, 0.069, 0.257, 0.107),
        "type":  "text",
    },
    "area": {
        "label": "Area",
        "roi":   (0.298, 0.071, 0.388, 0.107),
        "type":  "text",
    },
    "trench": {
        "label": "Trench",
        # y1 expanded from 0.092 → 0.107 to give Apple Vision adequate pixel height
        "roi":   (0.447, 0.072, 0.609, 0.107),
        "type":  "text",
    },
    "context": {
        "label": "Context",
        "roi":   (0.678, 0.068, 0.948, 0.107),
        "type":  "text",
    },
    "date": {
        "label": "Date",
        # y1 expanded from 0.129 → 0.134 for safety margin
        "roi":   (0.117, 0.105, 0.380, 0.134),
        "type":  "text",
    },
    "recorded_by": {
        "label": "Recorded by",
        "roi":   (0.504, 0.105, 0.948, 0.134),
        "type":  "text",
    },
    "feature_type": {
        "label":   "Feature Type",
        # x0 pulled left to 0.206 to capture the Deposit checkbox column
        "roi":     (0.206, 0.139, 0.755, 0.164),
        "type":    "checkbox_group",
        "options": ["Deposit", "Cut", "Fill", "Structural"],
    },
    "description": {
        "label": "Description",
        "roi":   (0.070, 0.172, 0.952, 0.496),
        "type":  "text",
    },
    "above": {
        "label": "Above",
        "roi":   (0.105, 0.501, 0.910, 0.525),
        "type":  "text",
    },
    # The "This context" sub-annotation: a context number written by the recorder
    # inside the Above row boxes to indicate stratigraphic self-equality.
    # Stored as above_note in output, NOT as a peer key in the main record.
    "_above_note_roi": {
        "label": "_above_note_roi",
        "roi":   (0.449, 0.531, 0.645, 0.562),
        "type":  "above_note_roi",
    },
    "below": {
        "label": "Below",
        "roi":   (0.105, 0.568, 0.910, 0.593),
        "type":  "text",
    },
    "comments": {
        "label": "Comments",
        "roi":   (0.058, 0.604, 0.951, 0.793),
        "type":  "text",
    },
    "finds": {
        "label":   "Finds",
        "roi":     (0.173, 0.807, 0.571, 0.827),
        "type":    "checkbox_group",
        "options": ["Pot", "Lithic", "Bone", "Metal", "Other"],
    },
    "small_finds": {
        "label": "Small Finds",
        "roi":   (0.171, 0.835, 0.934, 0.882),
        "type":  "text",
    },
    "samples": {
        "label": "Samples",
        "roi":   (0.152, 0.888, 0.947, 0.918),
        "type":  "text",
    },
    "plan": {
        "label": "Plan",
        "roi":   (0.115, 0.922, 0.239, 0.953),
        "type":  "text",
    },
    "section": {
        "label": "Section",
        "roi":   (0.312, 0.925, 0.501, 0.952),
        "type":  "text",
    },
    "photo": {
        "label": "Photo",
        "roi":   (0.559, 0.920, 0.946, 0.960),
        "type":  "text",
    },
}

# ── Schema projections used by parsers ───────────────────────────────────────

# Fields that produce top-level output keys (excludes internal ROI helpers)
OUTPUT_FIELDS = {
    k: v for k, v in FIELD_REGIONS.items()
    if not k.startswith("_")
}

# LLM output schema: key → expected type description
OUTPUT_SCHEMA = {
    key: "array of strings from the 'options' list"
    if meta["type"] == "checkbox_group"
    else "string"
    for key, meta in OUTPUT_FIELDS.items()
}

# Checkbox fields: key → ordered options list (left-to-right on the physical form)
CHECKBOX_HINTS = {
    key: meta["options"]
    for key, meta in OUTPUT_FIELDS.items()
    if meta["type"] == "checkbox_group"
}

# Fields that contain context numbers; used for numeric confidence flagging
NUMERIC_FIELDS = {"above", "below", "context", "samples"}

# CSV column ordering: provenance metadata first, then all output fields
CSV_METADATA_KEYS   = ["_source_file", "_device", "_engine"]
CSV_METADATA_LABELS = ["Source File",  "Device",  "Engine"]
CSV_FIELD_KEYS      = list(OUTPUT_FIELDS.keys()) + ["above_note"]
CSV_FIELD_LABELS    = [OUTPUT_FIELDS[k]["label"] for k in OUTPUT_FIELDS] + ["Above Note"]
