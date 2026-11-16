import csv

from src import jurnal
from src.jurnal import JurnalEvenimente
from src.setari import SetariPacient


def test_setarile_salvate_sunt_reincarcate(tmp_path):
    cale = tmp_path / "setari.json"
    SetariPacient(prag_gest=0.9, prag_alerta=12.0, activ_emotie=False).salveaza(cale)
    assert SetariPacient.incarca(cale) == SetariPacient(prag_gest=0.9, prag_alerta=12.0, activ_emotie=False)


def test_fisier_lipsa_da_valori_implicite(tmp_path):
    assert SetariPacient.incarca(tmp_path / "nu_exista.json") == SetariPacient()


def test_fisier_corupt_da_valori_implicite(tmp_path):
    cale = tmp_path / "setari.json"
    cale.write_text("{nu este json", encoding="utf-8")
    assert SetariPacient.incarca(cale) == SetariPacient()


def test_valorile_invalide_sunt_inlocuite_cu_implicite(tmp_path):
    cale = tmp_path / "setari.json"
    cale.write_text('{"prag_gest": "mare", "activ_gest": 1, "prag_alerta": 10, "camp_necunoscut": 5}',
                    encoding="utf-8")
    setari = SetariPacient.incarca(cale)
    assert setari.prag_gest == SetariPacient().prag_gest
    assert setari.activ_gest is True
    assert setari.prag_alerta == 10


def test_jurnalul_scrie_antet_si_evenimente(tmp_path):
    j = JurnalEvenimente(tmp_path / "jurnal")
    j.inregistreaza(jurnal.PORNIRE)
    j.inregistreaza(jurnal.COMANDA, "I am in pain. \nNeed painkillers.")

    fisiere = list((tmp_path / "jurnal").glob("jurnal_*.csv"))
    assert len(fisiere) == 1
    with open(fisiere[0], encoding="utf-8") as f:
        randuri = list(csv.reader(f))
    assert randuri[0] == ["ora", "eveniment", "detalii"]
    assert [r[1:] for r in randuri[1:]] == [[jurnal.PORNIRE, ""], [jurnal.COMANDA, "I am in pain.  Need painkillers."]]
