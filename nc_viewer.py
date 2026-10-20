import sys
import numpy as np
import xarray as xr

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QComboBox,
)

from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SeisCanvas(FigureCanvasQTAgg):

    def __init__(self):

        self.fig = Figure(figsize=(14, 6))

        self.ax_inline = self.fig.add_subplot(131)
        self.ax_xline = self.fig.add_subplot(132)
        self.ax_slice = self.fig.add_subplot(133)

        super().__init__(self.fig)

        self.fig.tight_layout()


class NCViewer(QWidget):

    def __init__(self, nc_file):

        super().__init__()

        self.ds = xr.open_dataset(nc_file)

        self.setWindowTitle("PWI NetCDF Viewer")
        self.resize(1600, 800)

        self.iline_vals = self.ds.iline.values
        self.xline_vals = self.ds.xline.values

        if "twt" in self.ds.coords:
            self.twt_vals = self.ds.twt.values
        else:
            self.twt_vals = None

        self.build_ui()
        self.update_views()

    # -------------------------------------------------

    def build_ui(self):

        layout = QVBoxLayout(self)

        top = QHBoxLayout()

        top.addWidget(QLabel("Variable"))

        self.combo = QComboBox()

        self.combo.addItems(list(self.ds.data_vars))

        self.combo.currentIndexChanged.connect(
            self.update_views
        )

        top.addWidget(self.combo)

        layout.addLayout(top)

        # ------------------------------------

        self.il_slider = QSlider(Qt.Horizontal)

        self.il_slider.setMinimum(0)
        self.il_slider.setMaximum(
            len(self.iline_vals) - 1
        )

        self.il_slider.valueChanged.connect(
            self.update_views
        )

        layout.addWidget(QLabel("Inline"))
        layout.addWidget(self.il_slider)

        # ------------------------------------

        self.xl_slider = QSlider(Qt.Horizontal)

        self.xl_slider.setMinimum(0)
        self.xl_slider.setMaximum(
            len(self.xline_vals) - 1
        )

        self.xl_slider.valueChanged.connect(
            self.update_views
        )

        layout.addWidget(QLabel("Crossline"))
        layout.addWidget(self.xl_slider)

        # ------------------------------------

        self.twt_slider = QSlider(Qt.Horizontal)

        if self.twt_vals is not None:

            self.twt_slider.setMinimum(0)
            self.twt_slider.setMaximum(
                len(self.twt_vals) - 1
            )

            self.twt_slider.valueChanged.connect(
                self.update_views
            )

        layout.addWidget(QLabel("TWT"))
        layout.addWidget(self.twt_slider)

        # ------------------------------------

        self.info_label = QLabel()

        layout.addWidget(self.info_label)

        self.canvas = SeisCanvas()

        layout.addWidget(self.canvas)

    # -------------------------------------------------

    def update_views(self):

        name = self.combo.currentText()

        da = self.ds[name]

        il_idx = self.il_slider.value()
        xl_idx = self.xl_slider.value()

        il = self.iline_vals[il_idx]
        xl = self.xline_vals[xl_idx]

        ax1 = self.canvas.ax_inline
        ax2 = self.canvas.ax_xline
        ax3 = self.canvas.ax_slice

        ax1.clear()
        ax2.clear()
        ax3.clear()

        # ==================================================
        # 3D cube
        # ==================================================

        if da.ndim == 3:

            twt_idx = self.twt_slider.value()

            twt = self.twt_vals[twt_idx]

            inline = da.sel(
                iline=il
            ).transpose("twt", "xline")

            xline = da.sel(
                xline=xl
            ).transpose("twt", "iline")

            tslice = da.sel(
                twt=twt,
                method="nearest"
            )

            vmin = np.nanpercentile(
                da.values, 2
            )

            vmax = np.nanpercentile(
                da.values, 98
            )

            ax1.imshow(
                inline,
                aspect="auto",
                origin="upper",
                vmin=vmin,
                vmax=vmax
            )

            ax1.set_title(
                f"Inline {il}"
            )

            ax2.imshow(
                xline,
                aspect="auto",
                origin="upper",
                vmin=vmin,
                vmax=vmax
            )

            ax2.set_title(
                f"Crossline {xl}"
            )

            ax3.imshow(
                tslice,
                aspect="auto",
                origin="lower",
                vmin=vmin,
                vmax=vmax
            )

            ax3.set_title(
                f"TWT {twt:.1f}"
            )

        # ==================================================
        # 2D map
        # ==================================================

        elif da.ndim == 2:

            img = da.values

            vmin = np.nanpercentile(
                img, 2
            )

            vmax = np.nanpercentile(
                img, 98
            )

            ax3.imshow(
                img,
                origin="lower",
                aspect="auto",
                vmin=vmin,
                vmax=vmax
            )

            ax3.set_title(name)

            ax1.text(
                0.5,
                0.5,
                "2D Variable",
                ha="center"
            )

            ax2.text(
                0.5,
                0.5,
                "2D Variable",
                ha="center"
            )

        self.info_label.setText(
            f"Variable : {name}    "
            f"Inline : {il}    "
            f"Crossline : {xl}"
        )

        self.canvas.draw_idle()


def main():

    app = QApplication(sys.argv)

    viewer = NCViewer(
        r"OUTPUT_AR1_1_TEST_1\AR1_EEI_PWI.nc"
    )

    viewer.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()