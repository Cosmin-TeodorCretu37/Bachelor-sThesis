import csv
import logging
from datetime import datetime

from src.config import DIR_JURNAL

log = logging.getLogger(__name__)

PORNIRE = "PORNIRE"
OPRIRE = "OPRIRE"
COMANDA = "COMANDA"
ANULARE = "ANULARE"
ALERTA_URGENTA = "ALERTA_URGENTA"
ALERTA_DISCONFORT = "ALERTA_DISCONFORT"


class JurnalEvenimente:
    def __init__(self, director=DIR_JURNAL):
        self.director = director

    def inregistreaza(self, tip, detaliu=""):
        acum = datetime.now()
        fisier = self.director / f"jurnal_{acum:%Y-%m-%d}.csv"
        try:
            self.director.mkdir(parents=True, exist_ok=True)
            fisier_nou = not fisier.exists()
            with open(fisier, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if fisier_nou:
                    writer.writerow(["ora", "eveniment", "detalii"])
                writer.writerow([acum.strftime("%H:%M:%S"), tip, detaliu.replace("\n", " ")])
        except OSError:
            log.exception("Evenimentul nu a putut fi scris in jurnal")
