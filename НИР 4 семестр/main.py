from inference import Inferencer

from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QComboBox, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import onnxruntime as ort
import soundfile as sf
import numpy as np

import sys
import os

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.gradient = ["green", "#208e00", "#459c00", "#72aa00", "#a4b800", "#c7b000", "#d58e00", "#e36500", "#f13600", "red"]

        layout = QVBoxLayout()

        self.comboBox = QComboBox()
        for file in os.listdir("."):
            if file.endswith(".onnx"):
                self.comboBox.addItem(file)
        
        layout.addWidget(self.comboBox)

        button = QPushButton("Select file")
        button.clicked.connect(self.onButtonClicked)
        layout.addWidget(button)

        self.label = QLabel()
        self.label.setFont(QFont('Times', 14)) 
        layout.addWidget(self.label)

        self.image = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self.image, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.image)

        layoutWidget = QWidget()
        layoutWidget.setLayout(layout)
        self.setCentralWidget(layoutWidget)

    def onButtonClicked(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select file", "", "Audio file (*.wav)")
        if fileName != "":
            model_path = self.comboBox.currentText()
            model = ort.InferenceSession(model_path)
            inferencer = Inferencer(model)
            sound_data, sample_rate = sf.read(fileName)
            scores = inferencer.inference(sound_data, sample_rate)

            self.label.setText(f"{os.path.basename(fileName)} is spoofing with probability {np.mean(scores):.2f}")

            proba_colors = np.array(scores, dtype=int)
            proba_colors[0] = int(scores[0] * len(self.gradient))
            for i in range(1, len(scores)):
                proba_colors[i] = int((scores[i - 1] + scores[i]) / 2 * len(self.gradient))
            
            proba_interval = sample_rate // inferencer.shift_delimeter
            
            self.image.axes.cla()
            self.image.axes.plot(np.arange(len(sound_data)), sound_data)
            for i in range(len(proba_colors)):
                self.image.axes.fill_between((i * proba_interval, (i + 1) * proba_interval), -1, 1, facecolor=self.gradient[proba_colors[i]], alpha=0.5)
            self.image.draw()

app = QApplication(sys.argv)
window = MainWindow()
window.setWindowTitle("Antispoofing")
window.show()
app.exec()