import json
from types import SimpleNamespace

import pytest

from src.config import EMOTION_LABELS, EMOTION_MODEL, FACE_TASK, GESTURE_LABELS, GESTURE_MODEL, HAND_TASK
from src.emotion_classifier import EmotionClassifier
from src.gesture_classifier import GestureClassifier
from src.splash_screen import GESTURE_GUIDE
from training.collect_gesture_data import GESTURES, INSTRUCTIONS, is_valid_gesture, raised_fingers


@pytest.mark.parametrize("path", [GESTURE_MODEL, GESTURE_LABELS, EMOTION_MODEL, EMOTION_LABELS, HAND_TASK, FACE_TASK])
def test_model_files_exist(path):
    assert path.is_file()


def test_every_gesture_has_a_command_instruction_and_guide_entry():
    with open(GESTURE_LABELS) as f:
        model_gestures = set(json.load(f))
    assert model_gestures == set(GestureClassifier.MEDICAL_COMMANDS) == set(GESTURES) == set(INSTRUCTIONS)
    assert model_gestures == {gesture for _, gesture, _ in GESTURE_GUIDE}
    assert {GestureClassifier.EMERGENCY_GESTURE, GestureClassifier.CANCEL_GESTURE} <= model_gestures


def test_every_emotion_has_a_name_state_and_color():
    with open(EMOTION_LABELS) as f:
        model_emotions = set(json.load(f))
    assert model_emotions == set(EmotionClassifier.EMOTION_NAMES) == set(EmotionClassifier.CLINICAL_STATES)
    assert set(EmotionClassifier.CLINICAL_STATES.values()) == set(EmotionClassifier.STATE_COLORS)
    assert set(EmotionClassifier.DISCOMFORT_STATES) <= set(EmotionClassifier.STATE_COLORS)


def hand_with_fingers(raised):
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    for tip, up in zip([8, 12, 16, 20], raised, strict=True):
        points[tip].y = 0.3 if up else 0.7
    return points


@pytest.mark.parametrize("gesture, fingers, expected", [
    ("fist", [False] * 4, True),
    ("fist", [True, False, False, False], False),
    ("open_palm", [True] * 4, True),
    ("one", [True, False, False, False], True),
    ("two", [True, True, False, False], True),
    ("two", [True, True, True, False], False),
    ("three", [True, True, True, False], True),
    ("ok", [False, True, True, True], True),
])
def test_gesture_validation_during_collection(gesture, fingers, expected):
    assert raised_fingers(hand_with_fingers(fingers)) == fingers
    assert is_valid_gesture(gesture, fingers) == expected
