from types import SimpleNamespace

import numpy as np
import pytest

from src.detectoare import DetectorFata, DetectorMana, normalizeaza_mana


def mana(puncte):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in puncte]


@pytest.fixture
def puncte_aleatoare():
    return np.random.default_rng(0).uniform(0, 1, (21, 3))


def test_incheietura_devine_originea(puncte_aleatoare):
    vector = normalizeaza_mana(mana(puncte_aleatoare))
    assert len(vector) == 63
    assert vector[:3] == [0.0, 0.0, 0.0]
    assert max(map(abs, vector)) == pytest.approx(1.0)


def test_normalizarea_nu_depinde_de_pozitie_si_scara(puncte_aleatoare):
    original = normalizeaza_mana(mana(puncte_aleatoare))
    mutat_si_scalat = normalizeaza_mana(mana(puncte_aleatoare * 0.5 + 0.2))
    assert mutat_si_scalat == pytest.approx(original)


def test_mana_degenerata_nu_imparte_la_zero():
    assert normalizeaza_mana(mana([(0.5, 0.5, 0.0)] * 21)) == [0.0] * 63


def test_detectoarele_nu_gasesc_nimic_pe_cadru_gol():
    cadru = np.zeros((480, 640, 3), np.uint8)
    for detector in (DetectorMana(), DetectorFata()):
        assert detector.detecteaza(cadru) is None
        detector.inchide()
