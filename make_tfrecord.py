import librosa
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random

# 获取浮点数组
def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# 获取整型数据
def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 把数据添加到TFRecord中
def data_example(data, label):
    feature = {
        'data': _float_feature(data),
        'label': _int64_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


# 开始创建tfrecord数据
def create_data_tfrecord(data_list_path, save_path):
    with open(data_list_path, 'r') as f:
        data = f.readlines()
    with tf.io.TFRecordWriter(save_path) as writer:
        for d in tqdm(data):
            try:
                path, label = d.replace('\n', '').split('\t')
                wav, sr = librosa.load(path, sr=48000)
                wav_output = []
                '''
                intervals = librosa.effects.split(wav, top_db=20) #返回有声音的部分，即裁剪静音部分
                for sliced in intervals:
                    wav_output.extend(wav[sliced[0]:sliced[1]])
                '''
                wav_output.extend(wav)
                # [可能需要修改参数] 音频长度 48000 * 秒数
                wav_len = int(48000 * 2)
                for i in range(3):
                    # 裁剪过长的音频，过短的补0
                    if len(wav_output) > wav_len:
                        l = len(wav_output) - wav_len
                        r = random.randint(0, l)
                        wav_output = wav_output[r:wav_len + r]
                    else:
                        wav_output.extend(np.zeros(shape=[wav_len - len(wav_output)], dtype=np.float32))
                    #wav_output = np.array(wav_output)
                    # 转成梅尔频谱
                    #data = librosa.feature.melspectrogram(y=np.array(wav_output), sr=sr, n_fft=512, hop_length=256).reshape(-1).tolist()

                    #转化为梅尔倒谱系数mfcc
                    melspec = librosa.feature.melspectrogram(y=np.array(wav_output), sr=sr, n_fft=512, hop_length=256)
                    logmelspec = librosa.power_to_db(melspec)
                    data = librosa.feature.mfcc(S=logmelspec, n_mfcc=40).reshape(-1).tolist()

                    if len(data) != 40 * 376: continue
                    tf_example = data_example(data, int(label))
                    writer.write(tf_example.SerializeToString())
                    if len(wav_output) <= wav_len:
                        break
            except Exception as e:
                print(e)


if __name__ == '__main__':
    create_data_tfrecord('my_dataset/train_list.txt', 'my_dataset/train.tfrecord')
    create_data_tfrecord('my_dataset/test_list.txt', 'my_dataset/test.tfrecord')
