import numpy as np
import pytest
import torch

import corruptions
from model import SimpleBaseline


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    base = np.linspace(40, 220, 96, dtype=np.float32)[None, :, None]
    return np.clip(base + rng.normal(0, 20, (64, 96, 3)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", corruptions.CORRUPTIONS)
@pytest.mark.parametrize("severity", [1, 2, 3])
def test_corruption_preserves_shape_dtype(img, name, severity):
    out = corruptions.apply(img, name, severity, seed=7, box=(10, 5, 60, 50))
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("name", corruptions.CORRUPTIONS)
def test_corruption_deterministic_given_seed(img, name):
    a = corruptions.apply(img, name, 2, seed=11, box=(10, 5, 60, 50))
    b = corruptions.apply(img, name, 2, seed=11, box=(10, 5, 60, 50))
    np.testing.assert_array_equal(a, b)


def test_clean_is_identity(img):
    assert corruptions.apply(img, "clean", 0, seed=0) is img


@pytest.mark.parametrize("name", ["gaussian_noise", "low_light", "jpeg", "occluder"])
def test_corruption_severity_monotonic(img, name):
    err = [np.abs(corruptions.apply(img, name, s, seed=3, box=(10, 5, 60, 50)).astype(float) - img).mean()
           for s in (1, 2, 3)]
    assert err[0] < err[1] < err[2]


def test_simple_baseline_heatmap_shape():
    torch.manual_seed(0)
    m = SimpleBaseline("resnet18", pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 3, 256, 192))
    assert y.shape == (2, 17, 64, 48)
