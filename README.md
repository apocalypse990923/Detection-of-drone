# 无人机声音识别
基于梅尔倒谱系数mfcc与卷积神经网络CNN，对无人机声音进行探测识别

record.py 录制音频素材
clip.py 裁剪为2s音频
make_data_list.py 创建数据列表
make_tfrecord.py 根据list将音频素材制作为tfrecord数据集
model.py 初始化卷积神经网络模型
model_train.py 训练模型
model_test.py 测试模型，包括检测率、虚警率等指标
Main_UI.py 用户交互界面，实时录音，绘制信号波形与频谱并给出识别预测
