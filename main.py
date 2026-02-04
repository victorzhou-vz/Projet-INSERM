import sys
import os
os.environ["QT_LOGGING_RULES"] = "qt.pdf.links=false"

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QAbstractItemView,
    QHBoxLayout, QVBoxLayout,
    QPushButton, QFileDialog, QLabel,
    QTableWidget, QTableWidgetItem, QTextEdit, QSplitter, QLineEdit,
    QTabWidget, QTabBar
)


import match_references
import mistralAnalysis


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

    def run(self):
        try:
            for idx, job in mistralAnalysis.verify_jobs_stream(self.jobs):
                if self.isInterruptionRequested():
                    return
                self.job_updated.emit(idx, job)
            self.done.emit()
        except Exception as e:
            self.error.emit(str(e))



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Citation Verification UI")
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

        self.main_pdf_line = QLineEdit()
        self.main_pdf_line.setPlaceholderText("Main PDF (document with ~~citations~~)")
        pick_main_btn = QPushButton("Pick Main PDF…")
        pick_main_btn.clicked.connect(self.pick_main_pdf)

        self.refs_dir_line = QLineEdit()
        self.refs_dir_line.setPlaceholderText("References folder (PDFs)")
        pick_refs_btn = QPushButton("Pick References Folder…")
        pick_refs_btn.clicked.connect(self.pick_refs_folder)

        self.btn_match = QPushButton("1) Match references")
        self.btn_match.clicked.connect(self.run_match)

        self.btn_verify = QPushButton("2) Verify with Mistral")
        self.btn_verify.clicked.connect(self.run_verify)
        self.btn_verify.setEnabled(False)

        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)

        left_layout.addWidget(QLabel("Inputs"))
        left_layout.addWidget(self.main_pdf_line)
        left_layout.addWidget(pick_main_btn)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.refs_dir_line)
        left_layout.addWidget(pick_refs_btn)
        left_layout.addSpacing(10)
        left_layout.addWidget(self.btn_match)
        left_layout.addWidget(self.btn_verify)
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

        detail_layout.addWidget(QLabel("Details (context + justification)"))
        detail_layout.addWidget(self.detail, 1)


        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.addWidget(self.tabs)        
        right_splitter.addWidget(self.detail_panel)
        right_splitter.setSizes([850, 450])

        root.addWidget(left_panel)
        root.addWidget(right_splitter, 1)


        self.setCentralWidget(central)


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
            self.status.setText("Error: Main PDF not found.")
            return
        if not os.path.isdir(refs_dir):
            self.status.setText("Error: References folder not found.")
            return

        self.status.setText("Matching references…")
        self.btn_match.setEnabled(False)
        self.btn_verify.setEnabled(False)

        self.worker_match = WorkerMatch(main_pdf, refs_dir)
        self.worker_match.finished_jobs.connect(self.on_jobs_ready)
        self.worker_match.error.connect(self.on_error)
        self.worker_match.start()

    def on_jobs_ready(self, jobs):
        self.jobs = jobs
        self.populate_table(jobs)
        self.status.setText(f"Matched {len(jobs)} citations. Ready to verify.")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(True)

    def run_verify(self):
        if not self.jobs:
            self.status.setText("No jobs to verify. Run matching first.")
            return

        self.status.setText("Verifying with Mistral… (streaming results)")
        self.btn_verify.setEnabled(False)
        self.btn_match.setEnabled(False)

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
        self.status.setText("Verification finished.")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(True)

    def on_error(self, msg):
        self.status.setText(f"Error: {msg}")
        self.btn_match.setEnabled(True)
        self.btn_verify.setEnabled(bool(self.jobs))

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

    def update_row(self, row, job):
        def set_item(col, text):
            it = QTableWidgetItem("" if text is None else str(text))
            it.setFlags(it.flags() ^ Qt.ItemIsEditable)
            self.table.setItem(row, col, it)

        set_item(0, job.get("id"))
        set_item(1, job.get("page"))
        set_item(2, job.get("status"))

        # PDF button in column 3
        pdf_name = job.get("pdf_filename", "") or ""
        btn = QPushButton(pdf_name if pdf_name else "—")
        btn.setEnabled(bool(pdf_name))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("text-align:left; padding:4px;")
        btn.clicked.connect(lambda _=False, r=row: self.open_pdf_for_row(r))
        self.table.setCellWidget(row, 3, btn)

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
            self.status.setText("No PDF for this row.")
            return

        refs_dir = self.refs_dir_line.text().strip()
        pdf_path = os.path.join(refs_dir, pdf_name)

        if not os.path.exists(pdf_path):
            self.status.setText(f"PDF not found: {pdf_path}")
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
