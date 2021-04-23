#裁剪并制作数据集
import os
import wave
from pydub import AudioSegment
import pyaudio
import uuid

# 按秒截取音频
def get_part_wav(sound, start_time, end_time, part_wav_path)
    save_path = os.path.dirname(part_wav_path) #即上一级路径
    if not os.path.exists(save_path)
        os.makedirs(save_path)
    start_time = int(start_time)  1000
    end_time = int(end_time)  1000
    word = sound[start_timeend_time]
    word.export(part_wav_path, format=wav)
    print('文件保存在：%s 时长：%f s' % (part_wav_path, len(word)1000))

def crop_wav(path, crop_len)
    for src_wav_path in os.listdir(path)
        wave_path = os.path.join(path, src_wav_path)
        if wave_path[-4] != '.wav'
            continue
        else
            print(wave_path)
        file = wave.open(wave_path)
        # 采样总数
        a = file.getparams().nframes
        # 采样频率
        f = file.getparams().framerate
        # 获取音频时间长度
        t = a  f
        print('总时长为 %f s' % t)
        # 读取语音
        sound = AudioSegment.from_wav(wave_path)
        count_num = 0 #记录裁剪的个数
        for start_time in range(0, int(t), crop_len)
            save_path = os.path.join(path, os.path.basename(wave_path)[-4], str(uuid.uuid1()) + '.wav')
            get_part_wav(sound, start_time, start_time + crop_len, save_path)
            count_num += 1
        print('总共%d段裁剪音频' % count_num)

if __name__ == '__main__'
    crop_len = 2  #需裁剪的秒数
    crop_wav('save_audio', crop_len)
