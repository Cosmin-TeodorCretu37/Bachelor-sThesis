from types import SimpleNamespace

import numpy as np
import pytest

from src.detectors import FaceDetector, HandDetector, normalize_hand


def hand(points):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in points]


@pytest.fixture
def random_points():
    return np.random.default_rng(0).uniform(0, 1, (21, 3))


def test_wrist_becomes_the_origin(random_points):
    vector = normalize_hand(hand(random_points))
    assert len(vector) == 63
    assert vector[:3] == [0.0, 0.0, 0.0]
    assert max(map(abs, vector)) == pytest.approx(1.0)


def test_normalization_ignores_position_and_scale(random_points):
    original = normalize_hand(hand(random_points))
    moved_and_scaled = normalize_hand(hand(random_points * 0.5 + 0.2))
    assert moved_and_scaled == pytest.approx(original)


def test_degenerate_hand_does_not_divide_by_zero():
    assert normalize_hand(hand([(0.5, 0.5, 0.0)] * 21)) == [0.0] * 63


def test_detectors_find_nothing_on_an_empty_frame():
    frame = np.zeros((480, 640, 3), np.uint8)
    for detector in (HandDetector(), FaceDetector()):
        assert detector.detect(frame) is None
        detector.close()
