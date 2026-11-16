from src.monitorizare import ConfirmareGest, StareSustinuta


def simuleaza_gest(confirmare, gest_la_moment, durata, timp_confirmare=2.5, pas=0.1):
    confirmari = []
    for i in range(int(durata / pas)):
        t = round(i * pas, 1)
        _, confirmat = confirmare.actualizeaza(gest_la_moment(t), t, timp_confirmare)
        if confirmat:
            confirmari.append((t, confirmare.gest))
    return confirmari


def simuleaza_stare(stare, conditie_la_moment, durata, prag=7.0, pas=0.1):
    anunturi = []
    for i in range(int(durata / pas)):
        t = round(i * pas, 1)
        _, anunta = stare.actualizeaza(conditie_la_moment(t), t, prag)
        if anunta:
            anunturi.append(t)
    return anunturi


def test_gest_tinut_este_confirmat_o_singura_data():
    confirmari = simuleaza_gest(ConfirmareGest(grace=1.0), lambda t: "ok", 10)
    assert confirmari == [(2.5, "ok")]


def test_pierderea_scurta_a_mainii_nu_reseteaza_gestul():
    confirmari = simuleaza_gest(ConfirmareGest(grace=1.0), lambda t: None if 1.2 <= t < 1.7 else "ok", 6)
    assert confirmari == [(2.5, "ok")]


def test_pierderea_lunga_a_mainii_reseteaza_gestul():
    confirmari = simuleaza_gest(ConfirmareGest(grace=1.0), lambda t: None if 1.2 <= t < 2.5 else "ok", 6)
    assert confirmari == [(5.0, "ok")]


def test_schimbarea_gestului_reporneste_confirmarea():
    confirmari = simuleaza_gest(ConfirmareGest(grace=1.0), lambda t: "doi" if 3 <= t < 6 else "ok", 10)
    assert confirmari == [(2.5, "ok"), (5.5, "doi"), (8.5, "ok")]


def test_alerta_se_activeaza_dupa_prag_si_se_repeta():
    stare = StareSustinuta(interval_repetare=5, grace=1.0)
    anunturi = simuleaza_stare(stare, lambda t: t < 8 and not 3 <= t < 3.5, 20)
    assert anunturi[0] == 7.0
    assert len(anunturi) == 3
    assert stare.activa


def test_alerta_activa_ramane_pana_la_resetare():
    stare = StareSustinuta(interval_repetare=100)
    simuleaza_stare(stare, lambda t: t < 8, 12)
    assert stare.activa
    stare.reseteaza()
    assert not stare.activa
    assert stare.actualizeaza(False, 13, 7.0) == (0.0, False)


def test_starea_intrerupta_fara_grace_reporneste_cronometrul():
    anunturi = simuleaza_stare(StareSustinuta(interval_repetare=15), lambda t: t != 4.0, 13)
    assert anunturi == [11.1]


def test_progresul_creste_liniar():
    stare = StareSustinuta(interval_repetare=5)
    stare.actualizeaza(True, 0.0, 10.0)
    progres, anunta = stare.actualizeaza(True, 5.0, 10.0)
    assert progres == 0.5
    assert not anunta
