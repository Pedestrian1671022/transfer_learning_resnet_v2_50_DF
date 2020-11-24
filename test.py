import os
import tensorflow as tf
from tensorflow.contrib import slim
from resnet_utils import resnet_arg_scope
from resnet_v2_50 import resnet_v2_50

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

image_pixels = 224
classes = 2
test_size = 6371
batch_size = 5
tfrecord_file = "data_test.tfrecord"


def read_and_decode(serialized_example):
    features = tf.compat.v1.parse_single_example(serialized_example, features={
        "label": tf.compat.v1.FixedLenFeature([], tf.compat.v1.int64),
        "image_ir": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "image_rd": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
        "filename": tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img_ir = tf.compat.v1.decode_raw(features["image_ir"], tf.compat.v1.uint8)
    img_ir = tf.compat.v1.reshape(img_ir, [image_pixels, image_pixels, 3])
    img_ir = tf.compat.v1.cast(img_ir, tf.compat.v1.float32)

    img_rd = tf.compat.v1.decode_raw(features["image_rd"], tf.compat.v1.uint8)
    img_rd = tf.compat.v1.reshape(img_rd, [image_pixels, image_pixels, 3])
    img_rd = tf.compat.v1.cast(img_rd, tf.compat.v1.float32)

    label = tf.compat.v1.cast(features["label"], tf.compat.v1.int32)

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img_ir, img_rd, label, filename

images_ir = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input_ir")
images_rd = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input_rd")
labels = tf.compat.v1.placeholder(tf.compat.v1.int64, [None], name="input/y_input")

with slim.arg_scope(resnet_arg_scope()):
    logits, end_points = resnet_v2_50(images_ir, images_rd, num_classes=classes, is_training=False)

correct_prediction = tf.compat.v1.equal(labels, tf.compat.v1.argmax(end_points['predictions'], 1), name="correct_prediction")
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name="accuracy")

with tf.compat.v1.Session() as sess:
    ckpt = tf.compat.v1.train.get_checkpoint_state("ckpt_data")
    if ckpt:
        print(ckpt.model_checkpoint_path)
        tf.compat.v1.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('The ckpt file is None.')
    dataset_test = tf.compat.v1.data.TFRecordDataset([tfrecord_file])
    dataset_test = dataset_test.map(read_and_decode)
    dataset_test = dataset_test.repeat(1).shuffle(1000).batch(batch_size)
    iterator_test = dataset_test.make_initializable_iterator()
    next_element_test = iterator_test.get_next()
    sess.run(iterator_test.initializer)
    acc_sum = 0
    for _ in range(int(test_size/batch_size)):
        img_test_ir, img_test_rd, label_test, filename = sess.run(next_element_test)
        acc = sess.run(accuracy, feed_dict={images_ir: img_test_ir, images_rd: img_test_rd, labels: label_test})
        acc_sum += acc
    print("acc:", acc_sum/(int(test_size/batch_size)))