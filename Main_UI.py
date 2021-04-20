import sys
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
import numpy as np
import pyqtgraph as pg
import pyaudio
import time
import threading
import wave
import os
import librosa

class Example(QWidget):

    def __init__(self, chunk=512, channels=1, rate=48000, max_seconds=40, detect_seconds=2):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self.MAX_SECONDS = max_seconds
        self.DETECT_SECONDS = detect_seconds
        self._running = True
        self._frames = []
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(750,1200)
        self.plot_Time_Domain = PlotWidget(self)
        self.plot_Time_Domain.setGeometry(QtCore.QRect(25,25,550,550))

        self.plot_Frequence = PlotWidget(self)
        self.plot_Frequence.setGeometry(QtCore.QRect(25,600,550,550))

        start_button = QPushButton('开始录音', self)
        start_button.move(650, 200)
        start_button.clicked[bool].connect(self.start)

        start_button = QPushButton('停止录音', self)
        start_button.move(650, 500)
        start_button.clicked[bool].connect(self.stop)

        self.data1 = np.zeros(self.MAX_SECONDS*self.RATE, dtype=np.float32)
        self.curve1 = self.plot_Time_Domain.plot(self.data1, name="mode1")

        self.setWindowTitle('pyqtgraph example: Scrolling Plots')
        self.show()

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)

    def start(self):
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while(self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)
            if len(self._frames) > self.MAX_SECONDS * self.RATE / self.CHUNK:  #最多保存10s
                self._frames = self._frames[1:]
        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False

    def save(self, filename):
        p = pyaudio.PyAudio()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()

    def save_detect(self, filename):
        p = pyaudio.PyAudio()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames[-self.DETECT_SECONDS * self.RATE // self.CHUNK:]))
        wf.close()

    def update_data(self):
        self.save("show_audio.wav")
        self.save_detect("infer_audio.wav")
        wav, sr = librosa.load("show_audio.wav", sr=self.RATE)
        self.data1 = np.concatenate((self.data1, wav))
        self.data1 = self.data1[-self.MAX_SECONDS*self.RATE:]
        self.curve1.setData(self.data1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
