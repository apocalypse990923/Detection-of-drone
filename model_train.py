import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras

model = tf.keras.models.load_model('models/new_test.h5')

EPOCHS = 10
BATCH_SIZE=32

#读取并解析tfrecord数据
feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'data': tf.io.FixedLenFeature([15040], tf.float32),
    #'data': tf.io.FixedLenFeature([48128], tf.float32),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

def read_and_decode(example):
    feature_dict = tf.io.parse_single_example(example, feature_description)
    data = feature_dict['data']
    data = tf.reshape(data,[40, 376, 1])
    #data = tf.reshape(data,[128, 376, 1])
    label = feature_dict['label']
    label = tf.one_hot(label,2)
    return data, label

tfrecord_file = 'my_dataset/new_mfccs_train.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(read_and_decode)
train_dataset = dataset.shuffle(buffer_size=1000) \
        .repeat(count=EPOCHS) \
        .batch(batch_size=BATCH_SIZE) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

tfrecord_file = 'my_dataset/new_mfccs_test.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(read_and_decode)
dataset = dataset.shuffle(buffer_size = 100)
test_dataset = dataset.batch(batch_size = 32)

#loss, accuracy = model.evaluate(dataset)

#训练模型
model.fit(train_dataset, validation_data=test_dataset)
model.save(filepath='models/new_test.h5')
