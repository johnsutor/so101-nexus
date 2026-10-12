"""Rendering profiles resolve once and survive recording camera resizing."""

from dataclasses import FrozenInstanceError, asdict

import pytest

from so101_nexus import OverheadCamera, RenderConfig, RenderingQuality, TouchConfig, WristCamera
from so101_nexus.teleop.session import _build_recording_config


@pytest.mark.parametrize(
    "profile,samples,shadows", [("default", 8, True), ("reproducible", 4, False)]
)
def test_quality_profiles_resolve_explicit_settings(profile, samples, shadows):
    quality = RenderingQuality(profile)
    assert asdict(quality) == {
        "profile": profile,
        "msaa_samples": samples,
        "shadows": shadows,
        "shadow_map_size": 8192,
    }
    with pytest.raises(FrozenInstanceError):
        quality.msaa_samples = 0


def test_explicit_quality_overrides_profile_including_zero_and_false():
    quality = RenderingQuality(msaa_samples=0, shadows=False, shadow_map_size=1024)
    assert (quality.msaa_samples, quality.shadows, quality.shadow_map_size) == (0, False, 1024)
    assert RenderingQuality("reproducible", shadows=True).shadows is True


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"profile": "unknown"}, "profile"),
        ({"profile": None}, "profile"),
        ({"msaa_samples": -1}, "msaa_samples"),
        ({"msaa_samples": True}, "msaa_samples"),
        ({"msaa_samples": 4.0}, "msaa_samples"),
        ({"msaa_samples": "4"}, "msaa_samples"),
        ({"shadows": 0}, "shadows"),
        ({"shadows": "false"}, "shadows"),
        ({"shadow_map_size": 0}, "shadow_map_size"),
        ({"shadow_map_size": -1}, "shadow_map_size"),
        ({"shadow_map_size": True}, "shadow_map_size"),
        ({"shadow_map_size": 1024.0}, "shadow_map_size"),
    ],
)
def test_quality_rejects_invalid_values(kwargs, field):
    with pytest.raises(ValueError, match=field):
        RenderingQuality(**kwargs)


def test_render_defaults_resolve_quality_and_camera_defaults_inherit():
    assert RenderConfig().quality == RenderingQuality()
    assert RenderConfig(quality=None).quality == RenderingQuality()
    assert WristCamera().quality is None
    assert OverheadCamera().quality is None
    quality = RenderingQuality("reproducible")
    assert RenderConfig(quality=quality).quality is quality
    assert WristCamera(quality=quality).quality is quality
    assert OverheadCamera(quality=quality).quality is quality


@pytest.mark.parametrize("constructor", [RenderConfig, WristCamera, OverheadCamera])
def test_quality_settings_reject_untyped_dicts(constructor):
    with pytest.raises(TypeError, match="RenderingQuality"):
        constructor(quality={"shadows": False})


def test_recording_camera_resize_preserves_quality_and_modalities():
    quality = RenderingQuality("reproducible")
    original = TouchConfig(
        render=RenderConfig(quality=quality),
        observations=[
            WristCamera(quality=quality, modalities=("rgb", "depth")),
            OverheadCamera(quality=quality, modalities=("depth",)),
        ],
    )
    config = _build_recording_config(original, (32, 24), (64, 48))
    assert config.render.quality is quality
    cameras = config.observations
    assert [camera.quality for camera in cameras] == [quality, quality]
    assert [camera.modalities for camera in cameras] == [("rgb", "depth"), ("depth",)]
    assert [(camera.width, camera.height) for camera in cameras] == [(32, 24), (64, 48)]
    assert original.observations[0].width == 640
