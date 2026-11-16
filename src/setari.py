import json
import logging
from dataclasses import asdict, dataclass, fields

from src.config import FISIER_SETARI, PRAG_ALERTA, PRAG_INCREDERE_GEST, TIMP_CONFIRMARE_GEST

log = logging.getLogger(__name__)


def _valoare_valida(valoare, implicita):
    if isinstance(implicita, bool):
        return isinstance(valoare, bool)
    return isinstance(valoare, (int, float)) and not isinstance(valoare, bool)


@dataclass
class SetariPacient:
    prag_gest: float = PRAG_INCREDERE_GEST
    timp_confirmare: float = TIMP_CONFIRMARE_GEST
    prag_alerta: float = PRAG_ALERTA
    activ_gest: bool = True
    activ_emotie: bool = True
    activ_alerte: bool = True
    activ_istoric: bool = True

    @classmethod
    def incarca(cls, cale=FISIER_SETARI):
        try:
            with open(cale, encoding="utf-8") as f:
                date = json.load(f)
            implicite = cls()
            valori = {}
            for camp in fields(cls):
                implicita = getattr(implicite, camp.name)
                valoare = date.get(camp.name, implicita)
                valori[camp.name] = valoare if _valoare_valida(valoare, implicita) else implicita
            return cls(**valori)
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, AttributeError):
            log.warning("Fisierul de setari nu a putut fi citit, se folosesc valorile implicite.")
            return cls()

    def salveaza(self, cale=FISIER_SETARI):
        try:
            with open(cale, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=2)
        except OSError:
            log.exception("Setarile nu au putut fi salvate")
