import json

import numpy as np
import pytest

from src.config import EMOTION_LABELS
from src.emotion_classifier import EmotionClassifier


@pytest.fixture
def classifier():
    instance = EmotionClassifier.__new__(EmotionClassifier)
    with open(EMOTION_LABELS) as f:
        instance.index_to_emotion = {int(v): k for k, v in json.load(f).items()}
    return instance


def probabilities(classifier, **values):
    p = np.zeros(len(classifier.index_to_emotion))
    index = {e: i for i, e in classifier.index_to_emotion.items()}
    for emotion, value in values.items():
        p[index[emotion]] = value
    return p


def test_negative_emotions_above_threshold_mean_discomfort(classifier):
    emotion, confidence = classifier.choose_emotion(
        probabilities(classifier, neutral=0.6, sad=0.25, fear=0.1, happy=0.05), threshold=0.3)
    assert emotion == "sad"
    assert confidence == pytest.approx(0.35)


def test_negative_emotions_below_threshold_keep_the_most_likely_other_emotion(classifier):
    emotion, confidence = classifier.choose_emotion(
        probabilities(classifier, neutral=0.5, happy=0.25, angry=0.15, sad=0.1), threshold=0.3)
    assert emotion == "neutral"
    assert confidence == pytest.approx(0.5)


def test_clear_negative_emotion_is_kept(classifier):
    emotion, _ = classifier.choose_emotion(probabilities(classifier, angry=0.9, neutral=0.1), threshold=0.3)
    assert emotion == "angry"
