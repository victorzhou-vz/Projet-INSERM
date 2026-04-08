"""
voice_input.py
==============
  - WhisperLoader : charge le modèle Whisper en arrière-plan au démarrage.
  - VoiceWorker   : 1er clic → démarre, 2ème clic → stop + transcription.

Dépendances :
    pip install openai-whisper sounddevice
"""

import threading
import numpy as np
from PySide6.QtCore import QThread, Signal


class WhisperLoader(QThread):
    model_ready = Signal(object)
    error       = Signal(str)

    MODEL_SIZE = "base"   # tiny | base | small | medium

    def run(self):
        try:
            import whisper
            model = whisper.load_model(self.MODEL_SIZE)
            self.model_ready.emit(model)
        except ImportError:
            self.error.emit("Module 'openai-whisper' manquant. Lancez : pip install openai-whisper")
        except Exception as e:
            self.error.emit(f"Erreur chargement Whisper : {e}")


class VoiceWorker(QThread):
    """
    Enregistrement toggle :
      - start()  → ouvre le micro, émet recording_started()
      - stop()   → ferme le micro, transcrit, émet transcription_ready(str)

    Le thread reste actif entre les deux appels (il attend stop()).
    """
    recording_started   = Signal()
    transcription_ready = Signal(str)
    error               = Signal(str)

    SAMPLERATE = 16_000
    CHUNK_SIZE = 1_024
    LANGUAGE   = "fr"

    def __init__(self, model):
        super().__init__()
        self.model       = model
        self._chunks     = []
        self._stop_event = threading.Event()

    def stop(self):
        """Appelé par le 2ème clic — déclenche la fin de l'enregistrement."""
        self._stop_event.set()

    def run(self):
        try:
            import sounddevice as sd
        except ImportError:
            self.error.emit("Module 'sounddevice' manquant. Lancez : pip install sounddevice")
            return

        self._chunks.clear()
        self._stop_event.clear()

        def _callback(indata, frames, time_info, status):
            self._chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self.SAMPLERATE,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
                callback=_callback,
            ):
                self.recording_started.emit()
                self._stop_event.wait()   # attend le 2ème clic

        except Exception as e:
            self.error.emit(f"Erreur enregistrement : {e}")
            return

        # Assemblage
        if not self._chunks:
            self.error.emit("Aucun audio capturé.")
            return

        audio_np = np.concatenate(self._chunks, axis=0).flatten()

        if np.abs(audio_np).max() < 1e-4:
            self.error.emit("Aucun son détecté. Vérifiez votre micro.")
            return

        # Transcription
        try:
            result = self.model.transcribe(audio_np, language=self.LANGUAGE, fp16=False)
            text   = result.get("text", "").strip()
        except Exception as e:
            self.error.emit(f"Erreur transcription : {e}")
            return

        if not text:
            self.error.emit("Transcription vide — réessayez en parlant plus fort.")
            return

        self.transcription_ready.emit(text)
