import logging
import queue
import threading

import pyttsx3

log = logging.getLogger(__name__)


class TextToSpeech:
    def __init__(self, rate=150):
        self.message_queue = queue.Queue()
        self.rate = rate

        self.worker = threading.Thread(target=self._process_messages, daemon=True)
        self.worker.start()

    @staticmethod
    def _find_english_voice(engine):
        for voice in engine.getProperty('voices'):
            languages = " ".join(str(language) for language in getattr(voice, 'languages', []) or []).lower()
            name = (voice.name or "").lower()
            voice_id = (voice.id or "").lower()
            if "english" in name or "en-" in voice_id or "en_" in voice_id or languages.startswith("en"):
                return voice.id
        return None

    def _process_messages(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        english_voice = None
        while True:
            text = self.message_queue.get()
            if text is None:
                break

            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', self.rate)
                if english_voice is None:
                    english_voice = self._find_english_voice(engine)
                if english_voice:
                    engine.setProperty('voice', english_voice)

                engine.say(text.replace("\n", " "))
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception:
                log.exception("Error while speaking the message")
            finally:
                self.message_queue.task_done()

    def say(self, text):
        self.message_queue.put(text)

    def say_urgent(self, text):
        with self.message_queue.mutex:
            self.message_queue.queue.clear()
        self.message_queue.put(text)

    def stop(self):
        with self.message_queue.mutex:
            self.message_queue.queue.clear()
        self.message_queue.put(None)
