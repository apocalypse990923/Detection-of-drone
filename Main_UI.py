import sys
from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication,QLabel,QWidget,QVBoxLayout
import numpy as np
import pyqtgraph as pg
import pyaudio
import time
import threading
import wave
import os
import librosa
from scipy.fft import fft, fftfreq
import random
import tensorflow as tf

class Example(QWidget):

    def __init__(self, chunk=512, channels=1, rate=48000, max_seconds=40, detect_seconds=2):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self.MAX_SECONDS = max_seconds
        self.DETECT_SECONDS = detect_seconds
        self._running = False
        self._frames = []
        self.model = tf.keras.models.load_model('models/mfccs_CNN.h5')
        self.result = "未检测到无人机"
        super().__init__()
        self.initUI()

    def initUI(self):
        self.resize(900,550)
        Label1 = QLabel(self)
        Label1.setText("时域波形图")
        Label1.move(200,25)

        self.plot_Time_Domain = PlotWidget(self)
        self.plot_Time_Domain.setGeometry(QtCore.QRect(25,50,400,400))

        Label2 = QLabel(self)
        Label2.setText("频谱图")
        Label2.move(650,25)

        self.plot_Frequence = PlotWidget(self)
        self.plot_Frequence.setGeometry(QtCore.QRect(450,50,400,400))

        start_button = QPushButton('开始录音', self)
        start_button.move(150, 500)
        start_button.clicked[bool].connect(self.start)

        stop_button = QPushButton('停止录音', self)
        stop_button.move(250, 500)
        stop_button.clicked[bool].connect(self.stop)

        Label3 = QLabel(self)
        Label3.setText("检测结果: " + self.result)
        Label3.move(500,500)

        self.data1 = np.zeros(self.MAX_SECONDS*self.RATE, dtype=np.float32)
        self.curve1 = self.plot_Time_Domain.plot(self.data1)

        self.data2 = np.zeros(self.DETECT_SECONDS*self.RATE, dtype=np.float32)
        self.curve2 = self.plot_Frequence.plot(self.data2)

        self.setWindowTitle('无人机探测系统')
        self.show()

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)

    def start(self):
        self._running = True
        threading._start_new_thread(self.__recording, ())

    def __recording(self):
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
        if self._running == False:
            return
        self.save("show_audio.wav")
        self.save_detect("infer_audio.wav")
        wav, sr = librosa.load("show_audio.wav", sr=self.RATE)
        self.data1 = np.concatenate((self.data1, wav))
        self.data1 = self.data1[-self.MAX_SECONDS*self.RATE:]
        self.curve1.setData(self.data1)

        wav2, sr = librosa.load("infer_audio.wav", sr=self.RATE)
        if len(wav2) > 0 :
            wav_pre = librosa.effects.preemphasis(wav2)
            self.data2 = np.abs(fft(wav_pre)[0:len(wav_pre)//2])
        self.curve2.setData(self.data2)

        self.result = self.predict(wav2)

    def predict(self, wav):
        wav_output = []
        wav_output.extend(wav)
        wav_len = int(48000 * 2)
        if len(wav_output) > wav_len:
            l = len(wav_output) - wav_len
            r = random.randint(0, l)
            wav_output = wav_output[r:wav_len + r]
        else:
            wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))

        melspec = librosa.feature.melspectrogram(y=np.array(wav_output), sr=self.RATE, n_fft=512, hop_length=256)
        logmelspec = librosa.power_to_db(melspec)
        data = librosa.feature.mfcc(S=logmelspec, n_mfcc=40)
        data = data[np.newaxis, ..., np.newaxis]
        label = int(tf.argmax(self.model.predict(data), 1))
        if label==0:
            return "检测到无人机！"
        else:
            return "未检测到无人机"


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
