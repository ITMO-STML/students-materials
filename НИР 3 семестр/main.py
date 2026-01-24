from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QComboBox, QFileDialog, QVBoxLayout

import onnxruntime as ort
import soundfile as sf
import numpy as np

import sys
import os
import math

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
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
        layout.addWidget(self.label)

        layoutWidget = QWidget()
        layoutWidget.setLayout(layout)
        self.setCentralWidget(layoutWidget)

    def onButtonClicked(self):
        fileName, _ = QFileDialog.getOpenFileName(None, "Select file", "", "Audio file (*.wav)")
        if fileName != "":
            model_path = self.comboBox.currentText()
            model = ort.InferenceSession(model_path)
            sound_data, sample_rate = sf.read(fileName)

            data_len = len(sound_data)
            sample_len = sample_rate
            sample_shift = sample_rate // 2
            max_shift = (data_len - sample_len + sample_shift - 1) // sample_shift * sample_shift
            sound_samples = [sound_data[i:min(i + sample_len, data_len)].tolist() for i in range(0, max_shift + 1, sample_shift)]
            sound_samples[-1] = sound_samples[-1] + [0] * (sample_len - len(sound_samples[-1]))
            sound_samples = np.array(sound_samples, dtype = "float32")

            scores = []
            for sound_sample in sound_samples:
                score = model.run(None, {"waves": [sound_sample]})[0][0]

                score = 1 / (1 + math.exp(-(score[1] - score[0])))
                scores.append(score)

            self.label.setText(f"{os.path.basename(fileName)} is spoofing with probability {np.mean(scores):.2f}")

app = QApplication(sys.argv)
window = MainWindow()
window.setWindowTitle("Antispoofing")
window.show()
app.exec()