"""Prompt enrichment for 3D-friendly T2I output.

Downstream 3D quality (especially AniGen's rigging) depends heavily on the
conditioning image being: single subject, plain background, symmetric pose,
full silhouette, even lighting. We inject positive cues into the prompt and
pass a style-specific negative prompt to the T2I pipeline.

English-only output. HunyuanDiT has bilingual CLIP+mT5 encoders — English
tokens work natively, we just need to keep the prompt under ~75 CLIP tokens
and front-load the pose cue.
"""
import re

STYLE_PRESETS = {
    "character": {
        "label": "Character (rigged-friendly)",
        "aspect": "portrait",
        "suffix": (
            "3D character reference turnaround, standing straight upright facing camera, "
            "both arms fully extended horizontally to the sides at shoulder height, "
            "both legs straight and parallel shoulder-width apart, feet flat on ground "
            "pointing forward, symmetrical T-pose, stylized 3D cartoon style, full body "
            "visible head to feet, centered, plain white background, clean silhouette, "
            "even studio lighting"
        ),
        "negative": (
            "cross-legged, legs crossed, one leg in front of other, legs together, "
            "kneeling, sitting, crouching, squatting, leaning, hip out, contrapposto, "
            "one leg raised, knee bent, arms at sides, arms down, arms crossed, "
            "hands on hips, hands in pockets, holding object, action pose, dynamic pose, "
            "running, jumping, fighting stance, walking, mid-step, "
            "side view, back view, three-quarter view, profile, close-up, portrait crop, "
            "cropped legs, cropped arms, cropped head, half body, "
            "multiple characters, group, duplicate, twins, from above, from below, "
            "photorealistic skin, motion blur, blurry, low quality, deformed, "
            "extra limbs, missing limbs, fused fingers, text, logo, watermark, signature, border"
        ),
    },
    "creature": {
        "label": "Creature / animal",
        "aspect": "portrait",
        "suffix": (
            "single creature standing, symmetric pose, all legs planted, tail visible, "
            "facing camera, full body, stylized 3D cartoon, plain white background, "
            "clean silhouette, even lighting"
        ),
        "negative": (
            "running, jumping, leaping, pouncing, fighting, rearing up, curled up, "
            "lying down, side view, close-up, cropped, multiple animals, motion blur, "
            "blurry, deformed, text, watermark"
        ),
    },
    "object": {
        "label": "Object / prop",
        "aspect": "square",
        "suffix": (
            "single object, centered composition, three-quarter view, resting on flat surface, "
            "full object visible, plain white background, 3D product render, "
            "even studio lighting, sharp focus, clean silhouette"
        ),
        "negative": (
            "close-up, cropped, partial view, multiple objects, collection, "
            "motion blur, blurry, reflection, mirror, hands holding object, "
            "person, character, text, logo, watermark"
        ),
    },
    "stylized": {
        "label": "Stylized / low-poly",
        "aspect": "portrait",
        "suffix": (
            "T-pose, arms stretched horizontally, legs straight, facing camera, "
            "low-poly 3D render, flat colors, cel-shaded, full body, "
            "plain white background, clean silhouette"
        ),
        "negative": (
            "photorealistic, arms at sides, action pose, side view, close-up, "
            "cropped, multiple subjects, motion blur, blurry, text, watermark"
        ),
    },
    "realistic": {
        "label": "Realistic / PBR",
        "aspect": "portrait",
        "suffix": (
            "T-pose, arms stretched horizontally at shoulder height, legs straight, "
            "facing camera, photorealistic 3D render, PBR materials, full body, "
            "plain white background, even studio lighting, clean silhouette"
        ),
        "negative": (
            "arms at sides, action pose, side view, close-up, cropped, multiple subjects, "
            "motion blur, blurry, cartoon, painting, low detail, text, watermark"
        ),
    },
}

DEFAULT_STYLE = "character"

# Terms that reliably degrade 3D reconstruction. Stripped case-insensitive.
ANTI_3D_PATTERNS = [
    r"\bmotion blur\b",
    r"\baction shot\b",
    r"\bdynamic pose\b",
    r"\bmid[- ]?action\b",
    r"\bharsh shadow(s)?\b",
    r"\bdramatic shadow(s)?\b",
    r"\bclose[- ]?up\b",
    r"\bextreme close[- ]?up\b",
    r"\bblurry\b",
    r"\bdepth of field\b",
    r"\bbokeh\b",
    r"\bcrop(ped)?\b",
]


def strip_anti_3d(text: str) -> str:
    out = text
    for pat in ANTI_3D_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE)
    out = re.sub(r"\s*,\s*,+", ", ", out)
    out = re.sub(r"\s+", " ", out).strip().strip(",").strip()
    return out


def enrich_prompt(user_prompt: str, style: str = DEFAULT_STYLE) -> tuple[str, str, str, str]:
    """Return (positive_prompt, negative_prompt, style_key, aspect)."""
    cleaned = strip_anti_3d(user_prompt or "")
    if not cleaned:
        cleaned = "subject"
    if style not in STYLE_PRESETS:
        style = DEFAULT_STYLE
    preset = STYLE_PRESETS[style]
    positive = f"{cleaned}, {preset['suffix']}"
    negative = preset.get("negative", "blurry, low quality, deformed, text, watermark")
    aspect = preset.get("aspect", "square")
    return positive, negative, style, aspect


def list_styles() -> list[dict]:
    return [{"key": k, "label": v["label"]} for k, v in STYLE_PRESETS.items()]
