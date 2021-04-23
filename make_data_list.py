#创建自定义训练数据列表
#audio_path为音频文件路径，用户需要提前把音频数据集存放在my_dataset/audio目录下，每个文件夹存放一个类别的音频数据，如my_dataset/audio/鸟叫声/...
#生成的数据类别的格式为音频路径\t音频对应的类别标签
import os
import librosa

def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)

    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w')

    for i in range(len(audios)):
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound)
            t = librosa.get_duration(filename=sound_path)
            # [可能需要修改参数] 过滤时长过小的音频
            if t >= 2:
                if sound_sum % 5 == 0:
                    f_test.write('%s\t%d\n' % (sound_path, i))
                else:
                    f_train.write('%s\t%d\n' % (sound_path, i))
                sound_sum += 1
        print("Audio：%d/%d" % (i + 1, len(audios)))

    f_test.close()
    f_train.close()

if __name__ == '__main__':
    get_data_list('my_dataset/audio', 'my_dataset')
