import sys
import os

import json
from datetime import datetime

os.environ["QT_LOGGING_RULES"] = "qt.pdf.links=false"

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QWaitCondition, QRectF, Property, QPropertyAnimation, QByteArray, QSize
from PySide6.QtGui import QColor, QPainter, QIcon, QPixmap, QImage

from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QAbstractItemView,
    QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel, QCheckBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QLineEdit,
    QTabWidget, QTabBar, QToolButton, QMenu, QSizePolicy
)

import match_references
import mistralAnalysisAPI_rerank as mistralAnalysisAPI
import rerank_specified_references as reranker_spec

from mistralai.client import Mistral as MistralClient
from voice_input import WhisperLoader, VoiceWorker


if getattr(sys, 'frozen', False):
    DOSSIER_ACTUEL = os.path.dirname(sys.executable)
else:
    DOSSIER_ACTUEL = os.path.dirname(os.path.abspath(__file__))

CHEMIN_RAPPORT = os.path.join(DOSSIER_ACTUEL, "rapport_verifications_refs.md")


INSERM_THEME = {
    "primary": "#E64415",        # Rouge Inserm
    "primary_dark": "#C83A10",
    "primary_light": "#FCE1D9",  # rouge clair pour sélections / hover doux

    "accent": "#000091",         
    "accent_light": "#417DC4",

    "bg": "#F5F7FA",
    "panel": "#FFFFFF",

    "border": "#D9E2EC",
    "text": "#1F2933",
    "muted": "#6B7280",

    "success": "#00A95F",
    "warning": "#C3992A",
    "danger": "#E1000F"
}

class WorkerMatch(QThread):
    finished_jobs = Signal(list)
    error = Signal(str)

    def __init__(self, main_pdf_path: str, refs_dir: str):
        super().__init__()
        self.main_pdf_path = main_pdf_path
        self.refs_dir = refs_dir

    def run(self):
        try:
            jobs = match_references.build_jobs(self.main_pdf_path, self.refs_dir)
            self.finished_jobs.emit(jobs)
        except Exception as e:
            self.error.emit(str(e))


class WorkerVerify(QThread):
    job_updated = Signal(int, dict)
    error = Signal(str)
    done = Signal()

    def __init__(self, jobs: list[dict], use_ollama: bool = False):
        super().__init__()
        self.jobs = jobs
        self.use_ollama = use_ollama

        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()
        self._paused = False
        self._stop_requested = False

    def pause(self) -> None:
        self._mutex.lock()
        try:
            self._paused = True
        finally:
            self._mutex.unlock()

    def resume(self) -> None:
        self._mutex.lock()
        try:
            self._paused = False
            self._pause_cond.wakeAll()
        finally:
            self._mutex.unlock()

    def stop(self) -> None:
        self.requestInterruption()
        self._mutex.lock()
        try:
            self._stop_requested = True
            self._paused = False
            self._pause_cond.wakeAll()
        finally:
            self._mutex.unlock()

    def _wait_if_paused(self) -> None:
        self._mutex.lock()
        try:
            while self._paused and not self._stop_requested and not self.isInterruptionRequested():
                self._pause_cond.wait(self._mutex)
        finally:
            self._mutex.unlock()

    def _should_abort(self) -> bool:
        self._mutex.lock()
        try:
            return self._stop_requested or self.isInterruptionRequested()
        finally:
            self._mutex.unlock()

    def run(self):
        try:
            for idx, job in mistralAnalysisAPI.verify_jobs_stream(self.jobs, should_abort=self._should_abort, use_ollama=self.use_ollama):
                self._wait_if_paused()
                if self._should_abort():
                    return
                self.job_updated.emit(idx, job)
            self.done.emit()

        except mistralAnalysisAPI.VerificationAborted:
            # Stop pressed
            return

        except Exception as e:
            self.error.emit(str(e))


class WorkerReVerify(QThread):
    job_updated = Signal(int, dict)
    error = Signal(str)
    done = Signal()

    def __init__(self, jobs: list[dict], job_indices: list[int], use_ollama: bool = False):
        super().__init__()
        self.jobs = jobs
        self.job_indices = job_indices
        self.use_ollama = use_ollama

        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()
        self._paused = False
        self._stop_requested = False

    def pause(self) -> None:
        self._mutex.lock()
        try:
            self._paused = True
        finally:
            self._mutex.unlock()

    def resume(self) -> None:
        self._mutex.lock()
        try:
            self._paused = False
            self._pause_cond.wakeAll()
        finally:
            self._mutex.unlock()

    def stop(self) -> None:
        self.requestInterruption()
        self._mutex.lock()
        try:
            self._stop_requested = True
            self._paused = False
            self._pause_cond.wakeAll()
        finally:
            self._mutex.unlock()

    def _wait_if_paused(self) -> None:
        self._mutex.lock()
        try:
            while self._paused and not self._stop_requested and not self.isInterruptionRequested():
                self._pause_cond.wait(self._mutex)
        finally:
            self._mutex.unlock()

    def _should_abort(self) -> bool:
        self._mutex.lock()
        try:
            return self._stop_requested or self.isInterruptionRequested()
        finally:
            self._mutex.unlock()

    def run(self):
        subset = [self.jobs[gi] for gi in self.job_indices]
        try:
            for local_idx, updated_job in reranker_spec.verify_jobs_stream(
                subset,
                should_abort=self._should_abort,
                use_ollama=self.use_ollama,
            ):
                self._wait_if_paused()
                if self._should_abort():
                    return
                global_idx = self.job_indices[local_idx]
                self.jobs[global_idx] = updated_job
                self.job_updated.emit(global_idx, updated_job)

            self.done.emit()

        except reranker_spec.VerificationAborted:
            return

        except Exception as e:
            self.error.emit(str(e))
            

class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 30)
        self._checked = False
        self._circle_position = 3.0

        self._animation = QPropertyAnimation(self, b"circle_position", self)
        self._animation.setDuration(180)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        if self._checked == checked:
            return
        self._checked = checked
        self.toggled.emit(self._checked)
        self._animation.stop()
        if checked:
            self._animation.setStartValue(self._circle_position)
            self._animation.setEndValue(float(self.width() - self.height() + 3))
        else:
            self._animation.setStartValue(self._circle_position)
            self._animation.setEndValue(3.0)
        self._animation.start()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setChecked(not self._checked)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setOpacity(0.4 if not self.isEnabled() else 1.0)
        bg_color = QColor(INSERM_THEME["primary"]) if self._checked else QColor("#C0C8D0")
        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(0, 0, self.width(), self.height(),
                                self.height() / 2, self.height() / 2)
        painter.setBrush(QColor("white"))
        painter.drawEllipse(QRectF(self._circle_position, 3,
                                   self.height() - 6, self.height() - 6))

    def getCirclePosition(self):
        return self._circle_position

    def setCirclePosition(self, pos):
        self._circle_position = pos
        self.update()

    circle_position = Property(float, getCirclePosition, setCirclePosition)


class CommandWorker(QThread):
    result_ready = Signal(str, list, float, int)   # action, indices (0-based), threshold, explain_index
    error = Signal(str)

    SYSTEM_PROMPT = """
    Tu es un interpréteur de commandes pour un outil de vérification de bibliographie scientifique.
    L'utilisateur te donne une instruction en langage naturel. Tu dois retourner UNIQUEMENT un objet JSON.

    Les indices dans le JSON sont TOUJOURS 0-based (numéro affiché - 1).

    Actions possibles :

    1. Re-vérifier des références précises :
    {"action": "re_verify", "indices": [0, 2, 4]}

    2. Re-vérifier les références dont le score est inférieur à un seuil :
    {"action": "re_verify_threshold", "threshold": 0.5}

    3. Re-vérifier toutes les références :
    {"action": "re_verify_all"}

    4. Expliquer la justification d'une référence précise :
    {"action": "explain", "index": 2}

    Exemples :
    "re-vérifie les références 2, 5 et 8" → {"action": "re_verify", "indices": [1, 4, 7]}
    "analyse plus en profondeur la ref 3" → {"action": "re_verify", "indices": [2]}
    "re-vérifie tout ce qui a un score sous 0.5" → {"action": "re_verify_threshold", "threshold": 0.5}
    "relance tout" → {"action": "re_verify_all"}
    "explique la justification de la référence 4" → {"action": "explain", "index": 3}
    "pourquoi la ref 7 a ce score ?" → {"action": "explain", "index": 6}

    Retourne UNIQUEMENT le JSON, sans texte ni balises markdown.
    """

    def __init__(self, client, user_text: str, jobs: list):
        super().__init__()
        self.client = client
        self.user_text = user_text
        self.jobs = jobs

    def run(self):
        try:
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": self.user_text}
                ],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            action = data.get("action", "")
            indices_raw = data.get("indices", [])
            threshold = float(data.get("threshold", 0.5))
            explain_index = int(data.get("index", -1))

            n = len(self.jobs)
            indices = [int(i) for i in indices_raw if 0 <= int(i) < n]

            self.result_ready.emit(action, indices, threshold, explain_index)

        except Exception as e:
            self.error.emit(str(e))


class ExplainWorker(QThread):
    """
    Reads the verification JSON, finds the job at `index`,
    and asks Mistral to produce a natural-language explanation
    based on citation context, retrieved chunks, and the stored justification.
    """
    explanation_ready = Signal(str)   # human-readable explanation
    error = Signal(str)

    JSON_PATH = "data/verification_results.json"

    SYSTEM_PROMPT = """
    Tu es un assistant expert en bibliographie scientifique.
    On te fournit les données brutes de vérification d'une citation :
    - La citation telle qu'elle apparaît dans l'article ("Citation")
    - Le contexte de la citation dans l'article ("Contexte")
    - Les extraits du PDF de référence jugés les plus pertinents ("Extraits source")
    - La justification brute produite par un modèle de vérification ("Justification brute")
    - Le score de cohérence attribué (de 0.1 à 1.0)

    Ta mission : rédiger en 3 à 5 phrases claires, en français, une explication pédagogique
    du score obtenu. Explique concrètement POURQUOI la citation correspond (ou ne correspond pas)
    aux extraits source. Mentionne les éléments précis qui ont justifié le score.
    Ne répète pas mot pour mot la justification brute : reformule, explique, synthétise.
    """

    def __init__(self, client, job: dict):
        super().__init__()
        self.client = client
        self.job = job

    def run(self):
        try:
            job = self.job

            # Build context for Mistral
            citation  = job.get("raw_citation", "N/A")
            context   = job.get("citation_context", "N/A")
            score     = job.get("mistral_score", "N/A")
            justif    = job.get("mistral_justification", "N/A")
            chunks    = job.get("retrieved_chunks", [])

            chunks_text = "\n\n---\n\n".join(
                c.get("text_preview", "") for c in chunks if c.get("text_preview")
            ) or "Aucun extrait disponible."

            user_prompt = (
                f"Citation : {citation}\n\n"
                f"Contexte : {context}\n\n"
                f"Extraits source :\n{chunks_text}\n\n"
                f"Justification brute : {justif}\n\n"
                f"Score : {score}"
            )

            # 3. Call Mistral
            response = self.client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ]
            )
            explanation = response.choices[0].message.content.strip()
            self.explanation_ready.emit(explanation)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.apply_theme()

        self.setWindowTitle("INSERM Reference Matcher & Verifier")
        self.resize(1200, 700)

        self.jobs = []

        _api_key = os.environ.get("MISTRAL_API_KEY", "")
        self._cmd_mistral = MistralClient(api_key=_api_key) if _api_key else None

        # Whisper model — chargé en arrière-plan au démarrage
        self._whisper_model = None
        self._whisper_loader = WhisperLoader()
        self._whisper_loader.model_ready.connect(self._on_whisper_ready)
        self._whisper_loader.error.connect(self._on_whisper_error)
        self._whisper_loader.start()

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---------- Left panel (inputs + actions)
        left_panel = QWidget()

        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        left_panel.setObjectName("leftPanel")
        left_panel.setStyleSheet(f"""
            QWidget#leftPanel {{
                background: {INSERM_THEME["panel"]};
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 14px;
            }}
        """)

        self.main_pdf_line = QLineEdit()
        self.main_pdf_line.setPlaceholderText("Main PDF (document with ~~citations~~)")
        self.main_pdf_line.setReadOnly(True)

        pick_main_btn = QPushButton("Pick Main PDF…")
        pick_main_btn.clicked.connect(self.pick_main_pdf)

        self.refs_dir_line = QLineEdit()
        self.refs_dir_line.setPlaceholderText("References folder (PDFs)")

        self.refs_dir_line.setReadOnly(True)
        pick_refs_btn = QPushButton("Pick References Folder…")
        pick_refs_btn.clicked.connect(self.pick_refs_folder)


        # --- Toggle Ollama / Mistral ---
        toggle_row = QWidget()
        toggle_row_layout = QHBoxLayout(toggle_row)
        toggle_row_layout.setContentsMargins(0, 0, 0, 0)
        toggle_row_layout.setSpacing(10)

        self.ollama_toggle = ToggleSwitch()
        self.ollama_toggle.toggled.connect(self.on_ollama_toggle)

        self.toggle_label = QLabel("Use Mistral API")
        self.toggle_label.setStyleSheet(f"color: {INSERM_THEME['text']}; font-size: 12px;")

        toggle_row_layout.addWidget(self.ollama_toggle)
        toggle_row_layout.addWidget(self.toggle_label)
        toggle_row_layout.addStretch()

        self.btn_match = QPushButton("1) Match references")
        self.btn_match.clicked.connect(self.run_match)

        self.btn_verify = QPushButton("2) Verify with Mistral")
        self.btn_verify.clicked.connect(self.run_verify)
        self.btn_verify.setEnabled(False)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_verify)
        self.btn_stop.setEnabled(False)

        self.status = QLabel("Ready.")
        self.status.setFixedHeight(60)
        self.set_status("Ready.", "normal")
        
        left_layout.addWidget(QLabel("Inputs"))
        left_layout.addWidget(self.main_pdf_line)
        left_layout.addWidget(pick_main_btn)

        left_layout.addSpacing(6)

        left_layout.addWidget(self.refs_dir_line)
        left_layout.addWidget(pick_refs_btn)

        left_layout.addSpacing(10)

        left_layout.addWidget(toggle_row)
        left_layout.addSpacing(6)

        left_layout.addWidget(self.btn_match)
        left_layout.addWidget(self.btn_verify)

        left_layout.addWidget(self.btn_pause)
        left_layout.addWidget(self.btn_stop)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Status"))
        left_layout.addWidget(self.status)

        left_layout.addSpacing(10)

        # --- Zone de commande naturelle ---
        # --- Chat conversation display ---
        chat_label = QLabel("Command history")
        chat_label.setStyleSheet(f"color: {INSERM_THEME['muted']}; font-size: 11px; font-weight: 600;")

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFixedHeight(160)
        self.chat_display.setStyleSheet(f"""
            QTextEdit {{
                background: {INSERM_THEME["bg"]};
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 10px;
                padding: 6px;
                font-size: 11px;
            }}
        """)

        self.cmd_input = QTextEdit()
        self.cmd_input.setPlaceholderText(
            'Ex: "re-vérifie les références 2, 5 et 8"\n'
            'ou "re-vérifie tout ce qui a un score sous 0.5"'
        )
        self.cmd_input.setFixedHeight(68)
        self.cmd_input.setStyleSheet(f"""
            QTextEdit {{
                background: {INSERM_THEME["panel"]};
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 10px;
                padding: 6px;
                font-size: 12px;
            }}
        """)

        self.btn_send_cmd = QPushButton("▶ Send command")
        self.btn_send_cmd.setFixedHeight(30)
        self.btn_send_cmd.clicked.connect(self.send_natural_command)
        self.btn_send_cmd.setEnabled(False)

        MIC_ICON_B64 = "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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA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NSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANBaNQ8zAAAgAElEQVSLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAAAABAjQjoAAD0IqADAAAAADUioAMA0IuADgAAAADUiIAOAEAvAjoAAAAAUCMCOgAAvQjoAAAAAECNCOgAAPQioAMAAAAANSKgAwDQi4AOAAAAANSIgA4AQC8COgAAAABQIwI6AAC9COgAAAAAQI0I6AAA9CKgAwAAAAA1IqADANCLgA4AAAAA1IiADgBALwI6AAAAAFAjAjoAAL0I6AAA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AAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACYJBToQ6BABwAAAAAAAJgkFOhDoEAHAAAAAAAAmCQU6EOgQAcAAAAAAACY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AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADM8v8egfSc9L0yKAAAAABJRU5ErkJggg=="

        _raw_pixmap = QPixmap()
        _raw_pixmap.loadFromData(QByteArray.fromBase64(MIC_ICON_B64.encode()))
        # Rendre le fond noir transparent (garder uniquement le logo blanc)
        _img = _raw_pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        for _y in range(_img.height()):
            for _x in range(_img.width()):
                _c = QColor(_img.pixel(_x, _y))
                if _c.red() < 40 and _c.green() < 40 and _c.blue() < 40:
                    _img.setPixel(_x, _y, QColor(0, 0, 0, 0).rgba())
        pixmap = QPixmap.fromImage(_img)
        self.btn_mic = QPushButton()
        self.btn_mic.setIcon(QIcon(pixmap))
        self.btn_mic.setIconSize(QSize(24, 24))
        self.btn_mic.setFixedSize(30, 30)
        self.btn_mic.setToolTip("Microphone")

        self.btn_mic.setEnabled(False)   # activé quand Whisper est prêt
        self.btn_mic.clicked.connect(self.start_voice_input)
        self.btn_mic.setStyleSheet(f"""
            QPushButton {{
                background-color: {INSERM_THEME["accent"]};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 16px;
                padding: 4px;
            }}
            QPushButton:hover  {{ background-color: {INSERM_THEME["accent_light"]}; }}
            QPushButton:disabled {{ background-color: #D9E2EC; color: #9AA5B1; }}
        """)

        send_row = QWidget()
        send_row_layout = QHBoxLayout(send_row)
        send_row_layout.setContentsMargins(0, 0, 0, 0)
        send_row_layout.setSpacing(6)
        send_row_layout.addWidget(self.btn_send_cmd, 1)
        send_row_layout.addWidget(self.btn_mic, 0)

        left_layout.addWidget(chat_label)
        left_layout.addWidget(self.chat_display)
        left_layout.addWidget(self.cmd_input)
        left_layout.addWidget(send_row)
        left_layout.addSpacing(10)


        # ---------- Results panel (table)
        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "ID", "Page", "Status", "PDF", "Match score", "Mistral score"
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self.on_row_selected)

        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: #F8FBFC;
            }}
        """)

        # (Optionnel) Si tu veux éviter "Results" en double : ne mets pas de QLabel ici
        table_layout.addWidget(self.table, 1)

        # ---------- Right panel = Tabs (Results + Details + PDFs)
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(self.on_tab_close_requested)

        # Tab 0: Results (non fermable)
        self.tabs.addTab(table_panel, "Results")

        # Enlever le bouton X sur l’onglet "Results"
        tab_bar = self.tabs.tabBar()
        tab_bar.setTabButton(0, QTabBar.ButtonPosition.RightSide, None)
        tab_bar.setTabButton(0, QTabBar.ButtonPosition.LeftSide, None)

        # ---------- Details panel (always visible)
        self.detail_panel = QWidget()
        detail_layout = QVBoxLayout(self.detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)

        self.detail = QTextEdit()
        self.detail.setReadOnly(True)

        self.detail.setStyleSheet(f"""
            QTextEdit {{
                background: {INSERM_THEME["panel"]};
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 10px;
                padding: 8px;
            }}
        """)

        detail_layout.addWidget(QLabel("Details (context + justification)"))
        detail_layout.addWidget(self.detail, 1)


        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.addWidget(self.tabs)        
        right_splitter.addWidget(self.detail_panel)
        right_splitter.setSizes([850, 450])

        root.addWidget(left_panel)
        root.addWidget(right_splitter, 1)


        self.setCentralWidget(central)

    def apply_theme(self):
        c = INSERM_THEME
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {c["bg"]};
                color: {c["text"]};
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 13px;
            }}

            QLabel {{
                color: {c["accent"]};
            }}

            QLineEdit, QTextEdit, QTableWidget, QTabWidget::pane {{
                background: {c["panel"]};
                border: 1px solid {c["border"]};
                border-radius: 10px;
                padding: 6px;
            }}

            QLineEdit:focus, QTextEdit:focus {{
                border: 2px solid {c["primary"]};
            }}

            QPushButton {{
                background-color: {c["primary"]};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 12px;
                font-weight: 600;
            }}

            QPushButton:hover {{
                background-color: {c["primary_dark"]};
            }}

            QPushButton:pressed {{
                background-color: {c["accent"]};
            }}
            
            QPushButton:disabled {{
                background-color: #D9E2EC;
                color: #4B5563;
                border: 1px solid #C7D2DA;
            }}

            QTableWidget {{
                gridline-color: {c["border"]};
                selection-background-color: {c["primary_light"]};
                selection-color: {c["accent"]};
            }}

            QHeaderView::section {{
                background-color: {c["primary"]};
                color: white;
                border: none;
                padding: 8px;
                font-weight: 700;
            }}

            QTabBar::tab {{
                background: #EAF1F4;
                color: {c["accent"]};
                padding: 8px 14px;
                margin-right: 4px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}

            QTabBar::tab:selected {{
                background: {c["panel"]};
                border: 1px solid {c["border"]};
                border-bottom-color: {c["panel"]};
            }}

            QTabBar::tab:hover {{
                background: {c["primary_light"]};
            }}

            QCheckBox {{
                spacing: 10px;
                color: #000091;
                font-weight: 600;
            }}

            QCheckBox::indicator {{
                width: 46px;
                height: 24px;
                border-radius: 12px;
                background: #D9E2EC;
                border: 1px solid #C7D2DA;
            }}

            QCheckBox::indicator:unchecked {{
                background: #D9E2EC;
            }}

            QCheckBox::indicator:checked {{
                background: #E64415;
            }}
        """)
    def set_status(self, text, kind="normal"):
        colors = {
            "normal": INSERM_THEME["accent"],
            "success": INSERM_THEME["success"],
            "warning": INSERM_THEME["warning"],
            "error": INSERM_THEME["danger"],
        }
        self.status.setText(text)
        self.status.setStyleSheet(f"""
            QLabel {{
                background: white;
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 10px;
                padding: 10px;
                color: {colors.get(kind, INSERM_THEME["accent"])};
                font-weight: 600;
            }}
        """)

    # --------- UI actions
    def on_tab_close_requested(self, index: int):
        # "Results" ne doit jamais être fermable
        if self.tabs.tabText(index) == "Results":
            return
        self.tabs.removeTab(index)  

    def pick_main_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select main PDF", "", "PDF (*.pdf)")
        if path:
            self.main_pdf_line.setText(path)

    def on_ollama_toggle(self, enabled: bool):
        if enabled:
            self.toggle_label.setText("Use Ollama (local)")
            self.toggle_label.setStyleSheet(f"color: {INSERM_THEME['primary']}; font-size: 12px; font-weight: 600;")
        else:
            self.toggle_label.setText("Use Mistral API")
            self.toggle_label.setStyleSheet(f"color: {INSERM_THEME['text']}; font-size: 12px;")

    def pick_refs_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select references folder", "")
        if path:
            self.refs_dir_line.setText(path)

    def run_match(self):
        main_pdf = self.main_pdf_line.text().strip()
        refs_dir = self.refs_dir_line.text().strip()

        if not os.path.exists(main_pdf):
            self.set_status("Error: Main PDF not found.", "error")
            return
        if not os.path.isdir(refs_dir):
            self.set_status("Error: References folder not found.", "error")
            return

        self.set_status("Matching references…", "normal")
        self.btn_match.setEnabled(False)
        self.btn_verify.setEnabled(False)

        self.worker_match = WorkerMatch(main_pdf, refs_dir)
        self.worker_match.finished_jobs.connect(self.on_jobs_ready)
        self.worker_match.error.connect(self.on_error)
        self.worker_match.start()

    def match_manually(self):
        # This function can be called from a context menu action to allow manual matching of a citation to a PDF.
        # For simplicity, it just opens a file dialog to pick a PDF, but you could make it more complex if needed.
        row = self.table.currentRow()
        if row < 0 or row >= len(self.jobs):
            self.set_status("No citation selected for manual matching.", "warning")
            return

        job = self.jobs[row]

        refs_dir = self.refs_dir_line.text().strip()
        initial_dir = refs_dir if os.path.isdir(refs_dir) else ""

        path, _ = QFileDialog.getOpenFileName(self, "Select PDF for manual matching", initial_dir, "PDF (*.pdf)")
        if path:
            pdf_name = os.path.basename(path)
            job["pdf_filename"] = pdf_name
            job["file_match_score"] = "Manual"
            self.update_row(row, job)
            self.set_status(f"Manually matched citation {job['id']} to {pdf_name}.", "success")

    def toggle_pause(self):
        if not getattr(self, "worker_verify", None) or not self.worker_verify.isRunning():
            return

        if self.btn_pause.text() == "Pause":
            self.worker_verify.pause()
            self.btn_pause.setText("Resume")
            self.set_status("Paused.", "warning")
        else:
            self.worker_verify.resume()
            self.btn_pause.setText("Pause")
            self.set_status("Verifying with Mistral… (resumed)", "normal")

    def stop_verify(self):
        if not getattr(self, "worker_verify", None) or not self.worker_verify.isRunning():
            return

        self.set_status("Stopping verification…", "warning")
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.worker_verify.stop()

    def on_jobs_ready(self, jobs):
        self.jobs = jobs
        self.populate_table(jobs)
        self.set_status(f"Matched {len(jobs)} citations. Ready to verify.", "success")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(True)

    def run_verify(self):
        if not self.jobs:
            self.set_status("No jobs to verify. Run matching first.", "warning")
            return

        self.verification_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        main_article_path = getattr(self, 'main_pdf_line', None)
        if main_article_path:
            path_text = main_article_path.text().strip()
            article_name = os.path.basename(path_text) if path_text else "Article_inconnu"
        else:
            article_name = "Article_inconnu"
            
        try:
            with open(CHEMIN_RAPPORT, "a", encoding="utf-8") as f:
                f.write(f"# Vérification lancée le {self.verification_start_time}\n\n")
                f.write(f"# Article concerné : {article_name}\n\n")
                #f.write("---\n\n")
        except Exception as e:
            print(f"Erreur init fichier auto : {e}")


        use_ollama = self.ollama_toggle.isChecked()
        mode_label = "Ollama (local)" if use_ollama else "Mistral API"
        self.set_status(f"Verifying with {mode_label}… (streaming results)", "normal")
        self.btn_verify.setEnabled(False)
        self.btn_match.setEnabled(False)
        self.ollama_toggle.setEnabled(False)  # verrouillage pendant la vérification

        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)

        self.worker_verify = WorkerVerify(self.jobs, use_ollama=use_ollama)
        self.worker_verify.job_updated.connect(self.on_job_updated)
        self.worker_verify.error.connect(self.on_error)
        self.worker_verify.done.connect(self.on_verify_done)
        self.worker_verify.start()

    def rerank_specified(self):
        if not self.jobs:
            self.set_status("No jobs to re-verify. Run matching first.", "warning")
            return

        selected_rows = sorted(set(idx.row() for idx in self.table.selectedIndexes()))
        if not selected_rows:
            selected_rows = list(range(len(self.jobs)))

        use_ollama = self.ollama_toggle.isChecked()
        n = len(selected_rows)
        mode_label = "Ollama (local)" if use_ollama else "Mistral API"
        self.set_status(f"Deep re-verification ({n} ref{'s' if n != 1 else ''}) via {mode_label}…", "normal")

        self._launch_re_verify(selected_rows)

    def _chat_append(self, html: str):
        """Append a line of HTML to the chat display and scroll to bottom."""
        self.chat_display.append(html)
        sb = self.chat_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _chat_user(self, text: str):
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._chat_append(
            f'<div style="text-align:right; margin: 4px 0;">'
            f'<span style="background:{INSERM_THEME["primary"]}; color:white; '
            f'border-radius:10px; padding:4px 10px; display:inline-block; font-size:11px;">'
            f'{escaped}</span></div>'
        )

    def _chat_app(self, text: str, kind: str = "normal"):
        colors = {
            "normal": INSERM_THEME["accent"],
            "success": INSERM_THEME["success"],
            "warning": INSERM_THEME["warning"],
            "error": INSERM_THEME["danger"],
        }
        color = colors.get(kind, INSERM_THEME["accent"])
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._chat_append(
            f'<div style="text-align:left; margin: 4px 0;">'
            f'<span style="background:#EEF2F5; color:{color}; '
            f'border-radius:10px; padding:4px 10px; display:inline-block; font-size:11px;">'
            f'🤖 {escaped}</span></div>'
        )

    # --------- Whisper / voice input

    def _on_whisper_ready(self, model):
        self._whisper_model = model
        self.btn_mic.setEnabled(True)
        self.btn_mic.setToolTip("Cliquer pour commencer l'enregistrement")
        self.set_status("Modèle vocal prêt.", "success")

    def _on_whisper_error(self, msg: str):
        self.btn_mic.setToolTip(f"Indisponible : {msg}")
        self.set_status(f"Whisper indisponible : {msg}", "warning")

    def start_voice_input(self):
        """1er clic → démarre  |  2ème clic → arrête + transcrit."""
        if self._whisper_model is None:
            return

        # --- 2ème clic : on arrête l'enregistrement en cours ---
        if getattr(self, "_voice_worker", None) and self._voice_worker.isRunning():
            self.btn_mic.setEnabled(False)
            # self.btn_mic.setText("🎤")
            self.btn_mic.setToolTip("Transcription en cours…")
            self.set_status("Transcription en cours…", "normal")
            self._voice_worker.stop()
            return

        # --- 1er clic : on démarre l'enregistrement ---
        self.btn_mic.setEnabled(False)
        self.btn_send_cmd.setEnabled(False)
        self.set_status("Ouverture du micro…", "normal")

        self._voice_worker = VoiceWorker(self._whisper_model)
        self._voice_worker.recording_started.connect(self._on_recording_started)
        self._voice_worker.transcription_ready.connect(self._on_transcription)
        self._voice_worker.error.connect(self._on_voice_error)
        self._voice_worker.start()

    def _on_recording_started(self):
        # self.btn_mic.setText("⏹")
        self.btn_mic.setToolTip("Cliquer pour arrêter l'enregistrement")
        self.btn_mic.setEnabled(True)   # réactive pour le 2ème clic
        self.set_status("🔴 Enregistrement en cours… (re-cliquer pour arrêter)", "normal")

    def _on_transcription(self, text: str):
        # self.btn_mic.setText("🎤")
        self.btn_mic.setEnabled(True)
        self.btn_mic.setToolTip("Cliquer pour commencer l'enregistrement")
        self.set_status("Transcription reçue.", "success")
        self.cmd_input.setPlainText(text)
        self.send_natural_command()

    def _on_voice_error(self, msg: str):
        # self.btn_mic.setText("🎤")
        self.btn_mic.setEnabled(True)
        self.btn_mic.setToolTip("Cliquer pour commencer l'enregistrement")
        self.btn_send_cmd.setEnabled(bool(self.jobs))
        self._chat_app(f"Erreur micro : {msg}", "error")
        self.set_status("Erreur enregistrement.", "error")

    def send_natural_command(self):
        text = self.cmd_input.toPlainText().strip()
        if not text:
            return
        if not self._cmd_mistral:
            self._chat_app("❌ MISTRAL_API_KEY non définie.", "error")
            return

        self._chat_user(text)
        self.cmd_input.clear()
        self.btn_send_cmd.setEnabled(False)
        self._chat_app("⏳ Interprétation en cours…", "normal")
        self.set_status("Interprétation de la commande…", "normal")

        self._cmd_worker = CommandWorker(self._cmd_mistral, text, self.jobs)
        self._cmd_worker.result_ready.connect(self.on_command_result)
        self._cmd_worker.error.connect(self.on_command_error)
        self._cmd_worker.start()

    def on_command_result(self, action: str, indices: list, threshold: float, explain_index: int):
        self.btn_send_cmd.setEnabled(True)

        if action == "explain":
            n = len(self.jobs)
            if explain_index < 0 or explain_index >= n:
                self._chat_app("Je n'ai pas trouvé la référence demandée.", "warning")
                return
            ref_num = explain_index + 1
            self._chat_app(f"J'analyse la référence {ref_num}, un instant…", "normal")
            self.set_status(f"Génération de l'explication pour la référence {ref_num}…", "normal")
            self._explain_worker = ExplainWorker(self._cmd_mistral, self.jobs[explain_index])
            self._explain_worker.explanation_ready.connect(
                lambda txt: self._chat_app(txt, "normal")
            )
            self._explain_worker.error.connect(
                lambda err: self._chat_app(f"Erreur lors de l'explication : {err}", "error")
            )
            self._explain_worker.finished.connect(
                lambda: self.set_status("Explication générée.", "success")
            )
            self._explain_worker.start()
            return

        if action == "re_verify":
            if not indices:
                msg = "⚠️ Aucun indice valide trouvé dans la réponse."
                self._chat_app(msg, "warning")
                self.set_status(msg, "warning")
                return
            label = f"✅ Re-vérification des refs : {[i+1 for i in indices]}"
            self._chat_app(label, "success")
            self.set_status(f"Re-vérification de {len(indices)} référence(s)…", "normal")
            self._launch_re_verify(indices)

        elif action == "re_verify_threshold":
            indices = [
                i for i, job in enumerate(self.jobs)
                if isinstance(job.get("mistral_score"), float) and job["mistral_score"] < threshold
            ]
            if not indices:
                msg = f"✅ Aucune référence sous le seuil {threshold}."
                self._chat_app(msg, "success")
                self.set_status(msg, "success")
                return
            label = f"✅ Re-vérification des refs sous {threshold} : {[i+1 for i in indices]}"
            self._chat_app(label, "success")
            self.set_status(f"Re-vérification de {len(indices)} référence(s) sous {threshold}…", "normal")
            self._launch_re_verify(indices)

        elif action == "re_verify_all":
            indices = list(range(len(self.jobs)))
            label = f"✅ Re-vérification de toutes les références ({len(indices)})"
            self._chat_app(label, "success")
            self.set_status(f"Re-vérification de toutes les {len(indices)} références…", "normal")
            self._launch_re_verify(indices)

        else:
            msg = f"⚠️ Action inconnue reçue : {action}"
            self._chat_app(msg, "warning")
            self.set_status(msg, "warning")

    def on_command_error(self, msg: str):
        self.btn_send_cmd.setEnabled(True)
        self._chat_app(f"❌ Erreur : {msg}", "error")
        self.set_status(f"Erreur commande : {msg}", "error")

    def _launch_re_verify(self, indices: list):
        """Factorisation : lance WorkerReVerify sur une liste d'indices."""
        use_ollama = self.ollama_toggle.isChecked()
        self.btn_verify.setEnabled(False)
        self.btn_send_cmd.setEnabled(False)
        self.btn_match.setEnabled(False)
        self.ollama_toggle.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)

        self.worker_verify = WorkerReVerify(self.jobs, indices, use_ollama=use_ollama)
        self.worker_verify.job_updated.connect(self.on_job_updated)
        self.worker_verify.error.connect(self.on_error)
        self.worker_verify.done.connect(self.on_verify_done)
        self.worker_verify.start()
        
    def on_job_updated(self, idx, job):
        # Update internal list
        self.jobs[idx] = job

        # Update row cells
        self.update_row(idx, job)

        # Always refresh the detail panel if this row is currently selected
        if self.table.currentRow() == idx:
            self.show_details(job)

        try:
            with open(CHEMIN_RAPPORT, "a", encoding="utf-8") as f:
                ref_name = job.get("raw_citation", "Référence inconnue").replace('\n', ' ')
                page = job.get("page", "Inconnue")
                score = job.get("mistral_score", "N/A")
                flag = "⚪️"
                if isinstance(score, (int, float)):
                    if score == 0.1:
                        flag = "🔴"
                    elif score == 0.5:
                        flag = "🟠"
                    elif score >= 0.8:
                        flag = "🟢"
                justification = job.get("mistral_justification", "N/A").replace('\n', ' ')
                
                f.write(f"### {flag} Référence : {ref_name}, Page : {page}\n\n")
                f.write(f"- *Score Mistral :* {score}\n")
                f.write(f"- *Justification :* {justification}\n\n")
                f.write("---\n\n")
        except Exception as e:
            print(f"Erreur écriture temps réel : {e}")


    def on_verify_done(self):
        # Persist in-memory jobs to disk so ExplainWorker (and external tools) are always up to date
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/verification_results.json", "w", encoding="utf-8") as f:
                json.dump(self.jobs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: could not save verification_results.json: {e}")

        self.set_status("Verification finished.", "success")
        self._chat_app("✅ Vérification terminée avec succès.", "success")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(True)
        self.ollama_toggle.setEnabled(True)

        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_send_cmd.setEnabled(True)
        self.btn_mic.setEnabled(True)

    def on_error(self, msg):
        self.set_status(f"Error: {msg}", "error")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(bool(self.jobs))
        self.ollama_toggle.setEnabled(True)

        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_send_cmd.setEnabled(bool(self.jobs))
        self.btn_mic.setEnabled(True)

    def closeEvent(self, event):
        # Ensure background threads are stopped before closing
        for worker in (getattr(self, "worker_match", None),
                    getattr(self, "worker_verify", None)):
            if worker is not None and worker.isRunning():
                worker.requestInterruption()
                worker.quit()
                worker.wait(2000)  # wait up to 2 seconds
        super().closeEvent(event)

        
    
    # --------- Table logic
    def populate_table(self, jobs):
        self.table.setRowCount(len(jobs))
        for i, job in enumerate(jobs):
            self.update_row(i, job)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, 420)
        
    def update_row(self, row, job):
        def set_item(col, text):
            it = QTableWidgetItem("" if text is None else str(text))
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, it)

        set_item(0, job.get("id"))
        set_item(1, job.get("page"))
        set_item(2, job.get("status"))

        # ----- PDF cell = [PDF button | ⋮ menu button]
        pdf_name = job.get("pdf_filename", "") or ""

        cell_widget = QWidget()
        cell_layout = QHBoxLayout(cell_widget)
        cell_layout.setContentsMargins(4, 2, 4, 2)
        cell_layout.setSpacing(4)

        pdf_btn = QPushButton(pdf_name if pdf_name else "—")
        pdf_btn.setEnabled(bool(pdf_name))
        pdf_btn.setCursor(Qt.PointingHandCursor)
        pdf_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        pdf_btn.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 6px 10px;
                border-radius: 8px;
                background-color: {INSERM_THEME["primary_light"]};
                color: {INSERM_THEME["accent"]};
                font-weight: 600;
                border: 1px solid {INSERM_THEME["border"]};
            }}
            QPushButton:hover {{
                background-color: #C7EEF0;
            }}
            QPushButton:disabled {{
                background-color: #EEF2F5;
                color: #9AA5B1;
                border: 1px solid {INSERM_THEME["border"]};
            }}
        """)
        pdf_btn.clicked.connect(lambda _=False, r=row: self.open_pdf_for_row(r))

        menu_btn = QToolButton()
        menu_btn.setText("⋮")
        menu_btn.setCursor(Qt.PointingHandCursor)
        menu_btn.setPopupMode(QToolButton.InstantPopup)
        menu_btn.setFixedWidth(28)
        menu_btn.setStyleSheet(f"""
            QToolButton {{
                background: transparent;
                border: 1px solid {INSERM_THEME["border"]};
                border-radius: 8px;
                padding: 2px;
                font-size: 14px;
                font-weight: 700;
                color: {INSERM_THEME["accent"]};
            }}
            QToolButton:hover {{
                background: {INSERM_THEME["primary_light"]};
            }}
        """)

        menu = QMenu(menu_btn)
        menu.setStyleSheet(f"""
            QMenu {{
                background: white;
                border: 1px solid {INSERM_THEME["border"]};
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 18px;
                border-radius: 6px;
            }}
            QMenu::item:selected {{
                background: {INSERM_THEME["primary_light"]};
                color: {INSERM_THEME["accent"]};
            }}
        """)

        action_open = menu.addAction("Open PDF")
        action_open.triggered.connect(lambda _=False, r=row: self.open_pdf_for_row(r))

        action_match_manually = menu.addAction("Match PDF manually")
        action_match_manually.triggered.connect(lambda _=False, j=job: self.match_manually())

        action_copy = menu.addAction("Copy PDF name")
        action_copy.triggered.connect(
            lambda _=False, name=pdf_name: QApplication.clipboard().setText(name)
        )

        menu_btn.setMenu(menu)

        cell_layout.addWidget(pdf_btn, 1)
        cell_layout.addWidget(menu_btn, 0)

        self.table.setCellWidget(row, 3, cell_widget)

        set_item(4, job.get("file_match_score", ""))

        score = job.get("mistral_score", None)
        if score is None or score == "":
            flag = ""
            set_item(5, "")
        elif score >= 0.8:
            set_item(5, f"🟢 {score}")
        elif score >= 0.5:
            set_item(5, f"🟡 {score}")
        else:
            set_item(5, f"🔴 {score}")


    def on_row_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.jobs):
            return
        self.show_details(self.jobs[row])

    def show_details(self, job):
        raw = job.get("raw_citation", "")
        ctx = job.get("citation_context", "")
        just = job.get("mistral_justification", "")
        pdf = job.get("pdf_filename", "N/A")
        score = job.get("mistral_score", None)

        if score is None:
            flag = "N/A"
        elif score >= 0.8:
            flag = "🟢"
        elif score >= 0.5:
            flag = "🟡"
        else:
            flag = "🔴"
        
        self.detail.setPlainText(
            f"PDF: {pdf}\n"
            f"Mistral score: {score}\n\n"
            f"Raw citation:\n{raw}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Justification:\n{just}\n"
        )


    def open_pdf_for_row(self, row: int):
        if row < 0 or row >= len(self.jobs):
            return

        job = self.jobs[row]
        pdf_name = job.get("pdf_filename", "")
        if not pdf_name:
            self.set_status("No PDF for this row.", "warning")
            return

        refs_dir = self.refs_dir_line.text().strip()
        pdf_path = os.path.join(refs_dir, pdf_name)

        if not os.path.exists(pdf_path):
            self.set_status(f"PDF not found: {pdf_path}", "error")
            return

        self.open_pdf_tab(pdf_path, title=pdf_name)


    def open_pdf_tab(self, pdf_path: str, title: str):
        # If already open, focus it
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == title:
                self.tabs.setCurrentIndex(i)
                return

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        doc = QPdfDocument(container)

        # IMPORTANT: load() returns QPdfDocument.Error, not Status
        load_err = doc.load(pdf_path)

        if load_err == QPdfDocument.Error.None_ and doc.status() == QPdfDocument.Status.Ready:
            view = QPdfView(container)
            view.setDocument(doc)

            # Afficher toutes les pages en scroll vertical (pas seulement page 1)
            try:
                view.setPageMode(QPdfView.PageMode.MultiPage)
                view.setZoomMode(QPdfView.ZoomMode.FitToWidth)
            except Exception as e:
                print("PDF view config failed:", e)

            view.setFocusPolicy(Qt.StrongFocus)


            # Keep references alive
            container._pdf_doc = doc
            container._pdf_view = view

            layout.addWidget(view, 1)
            view.setFocus()
            self.tabs.addTab(container, title)
            self.tabs.setCurrentWidget(container)
            return

        # If it failed, show something meaningful (no errorString() in your version)
        msg = QLabel(
            "Failed to load PDF.\n\n"
            f"Path:\n{pdf_path}\n\n"
            f"load() error: {load_err}\n"
            f"doc.status(): {doc.status()}"
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)

        self.tabs.addTab(container, title)
        self.tabs.setCurrentWidget(container)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
