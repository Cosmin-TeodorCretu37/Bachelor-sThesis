import json
from types import SimpleNamespace

import pytest

from antrenare.colectare_date import GESTURI, INSTRUCTIUNI, degete_ridicate, gest_valid
from src.clasificator_emotii import ClasificatorEmotii
from src.clasificator_gesturi import ClasificatorGesturi
from src.config import MAPARE_EMOTII, MAPARE_GESTURI, MODEL_EMOTII, MODEL_GESTURI, TASK_FATA, TASK_MANA


@pytest.mark.parametrize("cale", [MODEL_GESTURI, MAPARE_GESTURI, MODEL_EMOTII, MAPARE_EMOTII, TASK_MANA, TASK_FATA])
def test_fisierele_modelelor_exista(cale):
    assert cale.is_file()


def test_fiecare_gest_are_comanda_si_instructiune():
    with open(MAPARE_GESTURI) as f:
        gesturi_model = set(json.load(f))
    assert gesturi_model == set(ClasificatorGesturi.COMENZI_MEDICALE) == set(GESTURI) == set(INSTRUCTIUNI)
    assert {ClasificatorGesturi.GEST_URGENTA, ClasificatorGesturi.GEST_ANULARE} <= gesturi_model


def test_fiecare_emotie_are_traducere_stare_si_culoare():
    with open(MAPARE_EMOTII) as f:
        emotii_model = set(json.load(f))
    assert emotii_model == set(ClasificatorEmotii.EMOTII_RO) == set(ClasificatorEmotii.STARI_CLINICE)
    assert set(ClasificatorEmotii.STARI_CLINICE.values()) == set(ClasificatorEmotii.CULORI_STARI)
    assert set(ClasificatorEmotii.STARI_DISCONFORT) <= set(ClasificatorEmotii.CULORI_STARI)


def mana_cu_degete(ridicate):
    puncte = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    for varf, sus in zip([8, 12, 16, 20], ridicate, strict=True):
        puncte[varf].y = 0.3 if sus else 0.7
    return puncte


@pytest.mark.parametrize("gest, degete, asteptat", [
    ("pumn_inchis", [False] * 4, True),
    ("pumn_inchis", [True, False, False, False], False),
    ("palma_deschisa", [True] * 4, True),
    ("unu", [True, False, False, False], True),
    ("doi", [True, True, False, False], True),
    ("doi", [True, True, True, False], False),
    ("trei", [True, True, True, False], True),
    ("ok", [False, True, True, True], True),
])
def test_validarea_gesturilor_la_colectare(gest, degete, asteptat):
    assert degete_ridicate(mana_cu_degete(degete)) == degete
    assert gest_valid(gest, degete) == asteptat
