"""
segy_viewer.py
--------------
PyQt5 interactive SEG-Y viewer with inline, crossline and time-slice panels
controlled by sliders.

Install dependencies
--------------------
    pip install PyQt5 matplotlib segyio numpy

Run
---
    python segy_viewer.py
"""

import sys
import numpy as np
import segyio

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QSlider, QPushButton, QFileDialog,
    QStatusBar, QGroupBox, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPalette, QColor


# =============================================================================
# DARK THEME
# =============================================================================

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
QGroupBox {
    border: 1px solid #3a3a5c;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: bold;
    color: #a0a0c0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #2a2a4a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5b8dee;
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #5b8dee;
    border-radius: 3px;
}
QPushButton {
    background-color: #5b8dee;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #7aa3f5;
}
QPushButton:pressed {
    background-color: #3a6dd0;
}
QPushButton#loadBtn {
    background-color: #16213e;
    border: 2px solid #5b8dee;
    font-size: 14px;
    padding: 12px 30px;
}
QPushButton#loadBtn:hover {
    background-color: #1e2f5c;
}
QLabel {
    color: #c0c0d8;
}
QLabel#valueLabel {
    color: #5b8dee;
    font-weight: bold;
    font-size: 13px;
    min-width: 60px;
}
QLabel#titleLabel {
    color: #5b8dee;
    font-size: 22px;
    font-weight: bold;
    letter-spacing: 2px;
}
QLabel#infoLabel {
    color: #808098;
    font-size: 11px;
}
QStatusBar {
    background-color: #16213e;
    color: #808098;
}
QFrame#divider {
    color: #3a3a5c;
}
"""


# =============================================================================
# MATPLOTLIB CANVAS
# =============================================================================

class SeismicCanvas(FigureCanvas):

    def __init__(self, parent=None):
        self.fig = Figure(facecolor="#1a1a2e", tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = []
        self.images = []
        self._init_axes()

    def _init_axes(self):
        self.fig.clear()
        self.axes = []
        self.images = []

        titles   = ["Inline Slice", "Crossline Slice", "Time Slice"]
        xlabels  = ["Crossline",    "Inline",           "Inline"]
        ylabels  = ["TWT (ms)",     "TWT (ms)",          "Crossline"]

        for i in range(3):
            ax = self.fig.add_subplot(1, 3, i + 1)
            ax.set_facecolor("#0d0d1a")
            ax.set_title(titles[i],  color="#a0a0c0", fontsize=10, pad=8)
            ax.set_xlabel(xlabels[i], color="#606080", fontsize=9)
            ax.set_ylabel(ylabels[i], color="#606080", fontsize=9)
            ax.tick_params(colors="#606080", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5c")
            self.axes.append(ax)
            self.images.append(None)

        self.draw()

    def update_slice(self, idx, data, extent, title):
        ax  = self.axes[idx]
        vmax = np.nanpercentile(np.abs(data), 98)
        vmax = vmax if vmax > 0 else 1.0

        if self.images[idx] is None:
            im = ax.imshow(
                data, aspect="auto", cmap="rainbow",
                vmin=-vmax, vmax=vmax,
                extent=extent, origin="upper"
            )
            self.images[idx] = im
            self.fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04,
                              format="%.0f").ax.tick_params(
                                  colors="#606080", labelsize=7)
        else:
            im = self.images[idx]
            im.set_data(data)
            im.set_extent(extent)
            im.set_clim(-vmax, vmax)

        ax.set_title(title, color="#a0a0c0", fontsize=10, pad=8)
        self.draw_idle()


# =============================================================================
# SLIDER ROW WIDGET
# =============================================================================

class SliderRow(QWidget):

    def __init__(self, label, min_val, max_val, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        name = QLabel(label)
        name.setFixedWidth(100)
        name.setFont(QFont("Courier New", 10))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(min_val)
        self.slider.setMaximum(max_val)
        self.slider.setValue((min_val + max_val) // 2)
        self.slider.setTickPosition(QSlider.NoTicks)

        self.val_label = QLabel(str(self.slider.value()))
        self.val_label.setObjectName("valueLabel")
        self.val_label.setFixedWidth(70)
        self.val_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider.valueChanged.connect(
            lambda v: self.val_label.setText(str(v))
        )

        layout.addWidget(name)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.val_label)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class SEGYViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.seismic = None
        self.twt     = None
        self.inlines = None
        self.xlines  = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("SEG-Y Viewer")
        self.setMinimumSize(1200, 750)
        self.setStyleSheet(DARK_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 8)
        root.setSpacing(10)

        # ---- title bar ---------------------------------------------------
        title = QLabel("SEG-Y  VIEWER")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        root.addWidget(title)

        # ---- canvas ------------------------------------------------------
        self.canvas = SeismicCanvas()
        root.addWidget(self.canvas, 1)

        # ---- controls ----------------------------------------------------
        ctrl_box = QGroupBox("Navigation")
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setSpacing(8)

        # placeholder sliders (replaced on load)
        self.sl_il = SliderRow("Inline",     0, 0)
        self.sl_xl = SliderRow("Crossline",  0, 0)
        self.sl_t  = SliderRow("Time (ms)",  0, 0)

        self.sl_il.slider.valueChanged.connect(self._on_il)
        self.sl_xl.slider.valueChanged.connect(self._on_xl)
        self.sl_t.slider.valueChanged.connect(self._on_t)

        ctrl_layout.addWidget(self.sl_il)
        ctrl_layout.addWidget(self.sl_xl)
        ctrl_layout.addWidget(self.sl_t)

        # ---- bottom bar: info + load button ------------------------------
        bottom = QHBoxLayout()

        self.info_label = QLabel("No file loaded")
        self.info_label.setObjectName("infoLabel")

        load_btn = QPushButton("⬆  Load SEG-Y")
        load_btn.setObjectName("loadBtn")
        load_btn.setFixedWidth(180)
        load_btn.clicked.connect(self._load_file)

        bottom.addWidget(self.info_label, 1)
        bottom.addWidget(load_btn)

        root.addWidget(ctrl_box)
        root.addLayout(bottom)

        # ---- status bar --------------------------------------------------
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready — load a SEG-Y file to begin")

    # ------------------------------------------------------------------
    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SEG-Y file", "",
            "SEG-Y files (*.sgy *.segy);;All files (*)"
        )
        if not path:
            return

        self.status.showMessage(f"Loading  {path} …")
        QApplication.processEvents()

        try:
            with segyio.open(path, "r", ignore_geometry=False) as f:
                self.seismic = segyio.tools.cube(f).astype(np.float32)
                self.twt     = f.samples.copy()
                self.inlines = f.ilines.copy()
                self.xlines  = f.xlines.copy()
        except Exception as e:
            self.status.showMessage(f"Error: {e}")
            return

        n_il, n_xl, n_t = self.seismic.shape

        # rewire sliders
        for sl, arr in [(self.sl_il, self.inlines),
                        (self.sl_xl, self.xlines),
                        (self.sl_t,  self.twt)]:
            sl.slider.blockSignals(True)
            sl.slider.setMinimum(int(arr[0]))
            sl.slider.setMaximum(int(arr[-1]))
            sl.slider.setValue(int(arr[len(arr) // 2]))
            sl.slider.blockSignals(False)
            sl.val_label.setText(str(sl.slider.value()))

        self.canvas._init_axes()
        self.canvas.images = [None, None, None]

        fname = path.split("/")[-1].split("\\")[-1]
        self.info_label.setText(
            f"{fname}   |   {n_il} IL  ×  {n_xl} XL  ×  {n_t} samples   "
            f"|   TWT {self.twt[0]:.0f} – {self.twt[-1]:.0f} ms"
        )
        self.status.showMessage("Loaded — use sliders to navigate")

        self._refresh_all()

    # ------------------------------------------------------------------
    def _nearest_idx(self, arr, val):
        return int(np.argmin(np.abs(arr - val)))

    def _on_il(self, val):
        if self.seismic is None:
            return
        idx = self._nearest_idx(self.inlines, val)
        data = self.seismic[idx, :, :].T
        extent = [self.xlines[0], self.xlines[-1], self.twt[-1], self.twt[0]]
        self.canvas.update_slice(0, data, extent, f"Inline  {self.inlines[idx]}")

    def _on_xl(self, val):
        if self.seismic is None:
            return
        idx = self._nearest_idx(self.xlines, val)
        data = self.seismic[:, idx, :].T
        extent = [self.inlines[0], self.inlines[-1], self.twt[-1], self.twt[0]]
        self.canvas.update_slice(1, data, extent, f"Crossline  {self.xlines[idx]}")

    def _on_t(self, val):
        if self.seismic is None:
            return
        idx = self._nearest_idx(self.twt, val)
        data = self.seismic[:, :, idx].T
        extent = [self.inlines[0], self.inlines[-1], self.xlines[0], self.xlines[-1]]
        self.canvas.update_slice(2, data, extent, f"Time slice  {self.twt[idx]:.0f} ms")

    def _refresh_all(self):
        self._on_il(self.sl_il.slider.value())
        self._on_xl(self.sl_xl.slider.value())
        self._on_t(self.sl_t.slider.value())


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = SEGYViewer()
    win.show()
    sys.exit(app.exec_())