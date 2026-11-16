import threading
import time

from src.procesare_fundal import ProcesareFundal


def asteapta_rezultat(procesare, timeout=2.0):
    limita = time.monotonic() + timeout
    while time.monotonic() < limita:
        rezultat, nou = procesare.preia()
        if nou:
            return rezultat
        time.sleep(0.01)
    raise TimeoutError


def test_rezultatul_este_preluat_o_singura_data():
    procesare = ProcesareFundal(lambda x: x * 2)
    assert procesare.trimite(3)
    assert asteapta_rezultat(procesare) == 6
    assert procesare.preia() == (6, False)
    procesare.opreste()


def test_cadrele_trimise_cat_timp_e_ocupat_sunt_ignorate():
    eliberare = threading.Event()
    procesare = ProcesareFundal(lambda x: eliberare.wait() and x)
    assert procesare.trimite(1)
    time.sleep(0.05)
    assert not procesare.trimite(2)
    eliberare.set()
    assert asteapta_rezultat(procesare) == 1
    procesare.opreste()


def test_eroarea_din_functie_nu_opreste_firul():
    procesare = ProcesareFundal(lambda x: 1 / x)
    procesare.trimite(0)
    assert asteapta_rezultat(procesare) is None
    procesare.trimite(4)
    assert asteapta_rezultat(procesare) == 0.25
    procesare.opreste()


def test_oprirea_inchide_firul():
    procesare = ProcesareFundal(lambda x: x)
    procesare.opreste()
    assert not procesare._fir.is_alive()
