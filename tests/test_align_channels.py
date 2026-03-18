"""Unit tests for the align_channels_nhwc helper in poison_tester_ui/data/channel_utils.py."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poison_tester_ui.data.channel_utils import align_channels_nhwc  # noqa: E402


def _make_nhwc(n: int, h: int, w: int, c: int) -> np.ndarray:
    return np.random.rand(n, h, w, c).astype(np.float32)


class TestAlignChannelsNhwc:
    def test_no_op_when_channels_match(self):
        x = _make_nhwc(4, 8, 8, 3)
        out_train, out_val, out_test, err = align_channels_nhwc(x, x, x, 3)
        assert err is None
        assert out_train.shape == x.shape

    def test_rgb_to_grayscale(self):
        x = _make_nhwc(4, 8, 8, 3)
        out_train, out_val, out_test, err = align_channels_nhwc(x, x, x, 1)
        assert err is None
        assert out_train.shape == (4, 8, 8, 1)
        assert out_train.dtype == np.float32

    def test_grayscale_to_rgb(self):
        x = _make_nhwc(4, 8, 8, 1)
        out_train, out_val, out_test, err = align_channels_nhwc(x, x, x, 3)
        assert err is None
        assert out_train.shape == (4, 8, 8, 3)
        assert out_train.dtype == np.float32

    def test_unsupported_conversion_returns_error(self):
        x = _make_nhwc(4, 8, 8, 2)
        _, _, _, err = align_channels_nhwc(x, x, x, 4)
        assert err is not None
        assert "Auto-conversion" in err

    def test_rgb_to_grayscale_luminance_weights(self):
        # Pure red pixel: R=1, G=0, B=0 → luminance ≈ 0.2126
        x = np.zeros((1, 1, 1, 3), dtype=np.float32)
        x[0, 0, 0, 0] = 1.0  # red channel
        out, _, _, err = align_channels_nhwc(x, x, x, 1)
        assert err is None
        assert abs(float(out[0, 0, 0, 0]) - 0.2126) < 1e-4

    def test_grayscale_to_rgb_repeats_value(self):
        x = np.full((1, 2, 2, 1), 0.5, dtype=np.float32)
        out, _, _, err = align_channels_nhwc(x, x, x, 3)
        assert err is None
        np.testing.assert_allclose(out[0, 0, 0, :], [0.5, 0.5, 0.5])
