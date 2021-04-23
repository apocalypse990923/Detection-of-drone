import librosa
import numpy as np
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
from tqdm import tqdm
import random

# 读取音频数据
def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=48000)
    '''
    intervals = librosa.effects.split(wav, top_db=20) #返回有声音的部分，即裁剪静音部分
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    '''
    wav_output = []
    wav_output.extend(wav)
    #assert len(wav_output) >= 8000, "有效音频小于0.5s"
    wav_len = int(48000 * 2)
    if len(wav_output) > wav_len:
        l = len(wav_output) - wav_len
        r = random.randint(0, l)
        wav_output = wav_output[r:wav_len + r]
    else:
        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
    melspec = librosa.feature.melspectrogram(y=np.array(wav_output), sr=sr, n_fft=512, hop_length=256)
    #melspec = melspec[np.newaxis, ..., np.newaxis]
    logmelspec = librosa.power_to_db(melspec)
    data = librosa.feature.mfcc(S=logmelspec, n_mfcc=40)
    data = data[np.newaxis, ..., np.newaxis]  #reshape为神经网络的四维输入格式，相当于reshape((1, 40, 376, 1))

    #return melspec
    return data

#预测
def infer(audio_path):
    model = tf.keras.models.load_model('models/new_test.h5')
    data = load_data(audio_path)
    result = model.predict(data)
    lab = tf.argmax(result, 1)
    return lab

#检测率
def detection_rate(path):
    #path='save_audio/无人机室外-10米'
    sounds = os.listdir(path)
    correct_num=0
    wrong=[]
    for sound in tqdm(sounds):
        if sound[-4:] != '.wav':
            continue
        sound_path = os.path.join(path, sound)
        label = infer(sound_path)
        #print('音频：%s 的预测结果标签为：%d' % (sound_path, label))
        if label==0:
            correct_num+=1
        else:
            wrong.append(sound)
    print('样本总数为%d 正确检测的个数为%d 检测率为%f' % (len(sounds), correct_num, correct_num/len(sounds)))
    #print(wrong)

#虚警率
def false_alarm_rate(path):
#path='my_dataset/audio/not_drone'
    sounds = os.listdir(path)
    wrong_num=0
    wrong=[]
    for sound in tqdm(sounds):
        if sound[-4:] != '.wav':
            continue
        sound_path = os.path.join(path, sound)
        label = infer(sound_path)
            #print('音频：%s 的预测结果标签为：%d' % (sound_path, label))
        if label==0:
            wrong_num+=1
            wrong.append(sound)
    print('样本总数为%d 虚报的个数为%d 虚警率为%f' % (len(sounds), wrong_num, wrong_num/len(sounds)))
    #print(wrong)

if __name__ == '__main__':
    '''
    # 要预测的音频文件
    path = 'save_audio/无人机室内8/0f84731c-97ab-11eb-bef9-80fa5b45a5aa.wav'
    label = infer(path)
    print('音频：%s 的预测结果标签为：%d' % (path, label))
    '''
    path='save_audio/无人机室外-10米'
    detection_rate(path)
    path='my_dataset/audio/not_drone'
    false_alarm_rate(path)
