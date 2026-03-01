"""Fixtures and helpers for LLM-powered visual verification tests."""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Sequence

import numpy as np
import pytest

from so101_nexus_core.visualization import CameraView, compose_frame

VISUAL_TEST_MODEL_ENV = "VISUAL_TEST_MODEL"
VISUAL_TEST_API_BASE_ENV = "VISUAL_TEST_API_BASE"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"

VERIFICATION_PROMPT = """\
You are a quality assurance inspector for robot simulation environments.

You are reviewing rendered images from a simulation: {env_description}

Images provided (one per camera angle):
{camera_name_list}

Verify ALL of the following:
1. SCENE ELEMENTS: These elements must be visible: {expected_elements}
2. ROBOT INTEGRITY: The robot arm appears correctly assembled (connected joints, no floating parts)
3. RENDER QUALITY: The scene is properly lit and rendered \
(no all-black frames, no major artifacts, no missing textures)
4. CAMERA DIVERSITY: Each camera shows a meaningfully different perspective of the scene

Respond with EXACTLY this format: verdict first, then reasoning:
VERDICT: PASS
Explanation here in 2-3 sentences.

or:
VERDICT: FAIL
Explanation here in 2-3 sentences."""


def encode_image_base64(img: np.ndarray) -> str:
    """Encode a uint8 numpy image as a base64 PNG data URI."""
    from PIL import Image

    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def verify_scene(
    views: Sequence[CameraView],
    env_description: str,
    expected_elements: str,
    model: str | None = None,
    api_base: str | None = None,
) -> tuple[bool, str]:
    """Send camera views to an LLM for visual verification.

    Returns (passed, explanation).
    """
    import litellm

    model = model or os.environ.get(VISUAL_TEST_MODEL_ENV, DEFAULT_MODEL)
    api_base = api_base or os.environ.get(VISUAL_TEST_API_BASE_ENV)

    camera_name_list = "\n".join(f"- {v.name}" for v in views)
    prompt = VERIFICATION_PROMPT.format(
        env_description=env_description,
        camera_name_list=camera_name_list,
        expected_elements=expected_elements,
    )

    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": encode_image_base64(compose_frame(views))},
        }
    ]

    messages = [
        {
            "role": "user",
            "content": [
                *image_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    kwargs = {"model": model, "messages": messages, "max_tokens": 8192}
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.completion(**kwargs)
    msg = response.choices[0].message

    content = msg.content
    if not content:
        content = getattr(msg, "reasoning_content", None) or ""

    text = content.strip()
    if not text:
        return False, f"LLM returned empty content. Full response: {response}"

    match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    passed = match is not None and match.group(1).upper() == "PASS"
    return passed, text


def _get_model() -> str | None:
    """Resolve the configured model, returning None if unavailable."""
    model = os.environ.get(VISUAL_TEST_MODEL_ENV, DEFAULT_MODEL)
    provider = model.split("/")[0] if "/" in model else ""

    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    if os.environ.get(VISUAL_TEST_API_BASE_ENV):
        return model

    needed_key = key_map.get(provider)
    if needed_key and not os.environ.get(needed_key):
        return None

    if not needed_key:
        return None

    return model


@pytest.fixture
def visual_verifier():
    """Fixture that skips tests when no vision model is configured."""
    model = _get_model()
    if model is None:
        pytest.skip("No visual test model configured (set ANTHROPIC_API_KEY or VISUAL_TEST_MODEL)")
    return model
