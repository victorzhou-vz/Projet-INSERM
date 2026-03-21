import sys
import os
from turtle import color
os.environ["QT_LOGGING_RULES"] = "qt.pdf.links=false"

from PySide6.QtCore import Qt, QThread, Signal, QMutex, QWaitCondition

from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QAbstractItemView,
    QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QLineEdit,
    QTabWidget, QTabBar, QToolButton, QMenu, QSizePolicy
)

import match_references
import mistralAnalysisAPI_rerank as mistralAnalysisAPI


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

    def __init__(self, jobs: list[dict]):
        super().__init__()
        self.jobs = jobs

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
            for idx, job in mistralAnalysisAPI.verify_jobs_stream(self.jobs, should_abort=self._should_abort):
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



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.apply_theme()

        self.setWindowTitle("INSERM Reference Matcher & Verifier")
        self.resize(1200, 700)

        self.jobs = []

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---------- Left panel (inputs + actions)
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
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
        self.status.setWordWrap(True)
        self.set_status("Ready.", "normal")

        left_layout.addWidget(QLabel("Inputs"))
        left_layout.addWidget(self.main_pdf_line)
        left_layout.addWidget(pick_main_btn)

        left_layout.addSpacing(6)

        left_layout.addWidget(self.refs_dir_line)
        left_layout.addWidget(pick_refs_btn)

        left_layout.addSpacing(10)

        left_layout.addWidget(self.btn_match)
        left_layout.addWidget(self.btn_verify)
        left_layout.addWidget(self.btn_pause)
        left_layout.addWidget(self.btn_stop)

        left_layout.addSpacing(10)
        left_layout.addWidget(QLabel("Status"))
        left_layout.addWidget(self.status, 1)


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

        self.set_status("Verifying with Mistral… (streaming results)", "normal")
        self.btn_verify.setEnabled(False)
        self.btn_match.setEnabled(False)

        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)

        self.worker_verify = WorkerVerify(self.jobs)
        self.worker_verify.job_updated.connect(self.on_job_updated)
        self.worker_verify.error.connect(self.on_error)
        self.worker_verify.done.connect(self.on_verify_done)
        self.worker_verify.start()

    def on_job_updated(self, idx, job):
        # update internal list
        self.jobs[idx] = job

        # update row cells
        self.update_row(idx, job)

        # if the selected row is this one, refresh details
        selected = self.table.currentRow()
        if selected == idx:
            self.show_details(job)

    def on_verify_done(self):
        self.set_status("Verification finished.", "success")        
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(True)

        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)

    def on_error(self, msg):
        self.set_status(f"Error: {msg}", "error")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(bool(self.jobs))

        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)

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
        set_item(5, job.get("mistral_score", ""))

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
        score = job.get("mistral_score", "N/A")

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
