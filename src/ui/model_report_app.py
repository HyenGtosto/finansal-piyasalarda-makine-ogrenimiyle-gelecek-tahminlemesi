"""Simple desktop UI for running models and viewing report figures."""

from __future__ import annotations

import subprocess
import sys
import threading
import tkinter as tk
import os
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
DATASET_SCRIPT = PROJECT_ROOT / "scripts" / "run_model_dataset_pipeline.py"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "run_model_training.py"
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "run_model_result_plots.py"
MPL_CONFIG_DIR = PROJECT_ROOT / ".matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ModelReportApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Bitcoin Sentiment Model Reports")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.status_text = tk.StringVar(value="Ready")
        self.figure_canvas: FigureCanvasTkAgg | None = None
        self.current_figure: Figure | None = None

        self._show_start_screen()

    def _clear_screen(self) -> None:
        for child in self.winfo_children():
            child.destroy()

    def _show_start_screen(self) -> None:
        self._clear_screen()

        frame = ttk.Frame(self, padding=32)
        frame.pack(expand=True, fill="both")

        title = ttk.Label(
            frame,
            text="Bitcoin Sentiment ML Reports",
            font=("Segoe UI", 18, "bold"),
        )
        title.pack(pady=(0, 18))

        run_button = ttk.Button(
            frame,
            text="Run machine learning models",
            command=self._start_training,
        )
        run_button.pack()

        status = ttk.Label(frame, textvariable=self.status_text)
        status.pack(pady=(18, 0))

    def _start_training(self) -> None:
        self.status_text.set("Running models and regenerating report figures...")
        for child in self.winfo_children():
            self._set_children_state(child, "disabled")

        thread = threading.Thread(target=self._run_training_pipeline, daemon=True)
        thread.start()

    def _set_children_state(self, widget: tk.Widget, state: str) -> None:
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_children_state(child, state)

    def _run_training_pipeline(self) -> None:
        MPL_CONFIG_DIR.mkdir(exist_ok=True)
        environment = os.environ.copy()
        environment["MPLCONFIGDIR"] = str(MPL_CONFIG_DIR)
        commands = [
            [sys.executable, str(DATASET_SCRIPT)],
            [sys.executable, str(TRAINING_SCRIPT)],
            [sys.executable, str(PLOT_SCRIPT)],
        ]
        output_parts: list[str] = []

        for command in commands:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            output_parts.append(result.stdout)
            output_parts.append(result.stderr)
            if result.returncode != 0:
                error_text = "\n".join(part for part in output_parts if part.strip())
                self.after(0, lambda: self._show_training_error(error_text))
                return

        self.after(0, self._show_report_screen)

    def _show_training_error(self, error_text: str) -> None:
        self._show_start_screen()
        self.status_text.set("Training failed.")
        messagebox.showerror("Training failed", error_text or "No error output was returned.")

    def _show_report_screen(self) -> None:
        self._clear_screen()

        root = ttk.Frame(self)
        root.pack(expand=True, fill="both")

        sidebar = ttk.Frame(root, width=280, padding=12)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        content = ttk.Frame(root, padding=12)
        content.pack(side="right", expand=True, fill="both")

        ttk.Label(sidebar, text="Report Figures", font=("Segoe UI", 12, "bold")).pack(
            anchor="w",
            pady=(0, 10),
        )

        figures = sorted(FIGURE_DIR.glob("*.png"))
        if not figures:
            ttk.Label(content, text="No report figures found.").pack(expand=True)
            return

        for figure_path in figures:
            button = ttk.Button(
                sidebar,
                text=self._format_figure_name(figure_path),
                command=lambda path=figure_path: self._display_figure(path),
            )
            button.pack(fill="x", pady=3)

        self.plot_container = ttk.Frame(content)
        self.plot_container.pack(expand=True, fill="both")
        self._display_figure(figures[0])

    def _display_figure(self, figure_path: Path) -> None:
        for child in self.plot_container.winfo_children():
            child.destroy()

        image = mpimg.imread(figure_path)
        figure = Figure(figsize=(8, 6), dpi=100)
        axis = figure.add_subplot(111)
        axis.imshow(image)
        axis.set_title(self._format_figure_name(figure_path))
        axis.axis("off")
        figure.tight_layout()

        self.current_figure = figure
        self.figure_canvas = FigureCanvasTkAgg(figure, master=self.plot_container)
        self.figure_canvas.draw()
        widget = self.figure_canvas.get_tk_widget()
        widget.pack(expand=True, fill="both")

    @staticmethod
    def _format_figure_name(figure_path: Path) -> str:
        return figure_path.stem.replace("_", " ").title()


def main() -> None:
    app = ModelReportApp()
    app.mainloop()


if __name__ == "__main__":
    main()
