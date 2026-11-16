import queue
import threading

import pyttsx3


class SintezaVocala:
    def __init__(self, viteza=150):
        self.coada_mesaje = queue.Queue()
        self.viteza = viteza

        self.worker = threading.Thread(target=self._proceseaza_voce, daemon=True)
        self.worker.start()

    @staticmethod
    def _gaseste_voce_engleza(engine):
        for voce in engine.getProperty('voices'):
            limbi = " ".join(str(l) for l in getattr(voce, 'languages', []) or []).lower()
            nume = (voce.name or "").lower()
            id_voce = (voce.id or "").lower()
            if "english" in nume or "en-" in id_voce or "en_" in id_voce or limbi.startswith("en"):
                return voce.id
        return None

    def _proceseaza_voce(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        voce_engleza = None
        while True:
            text = self.coada_mesaje.get()
            if text is None:
                break

            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', self.viteza)
                if voce_engleza is None:
                    voce_engleza = self._gaseste_voce_engleza(engine)
                if voce_engleza:
                    engine.setProperty('voice', voce_engleza)

                engine.say(text.replace("\n", " "))
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS] Eroare la redarea mesajului: {e}")
            finally:
                self.coada_mesaje.task_done()

    def roteste(self, text_engleza):
        self.coada_mesaje.put(text_engleza)

    def roteste_prioritar(self, text_engleza):
        with self.coada_mesaje.mutex:
            self.coada_mesaje.queue.clear()
        self.coada_mesaje.put(text_engleza)

    def opreste(self):
        with self.coada_mesaje.mutex:
            self.coada_mesaje.queue.clear()
        self.coada_mesaje.put(None)
