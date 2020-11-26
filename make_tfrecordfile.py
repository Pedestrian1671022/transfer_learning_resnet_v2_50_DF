import os
import cv2
import random
import tensorflow as tf


dataset_dir = "data/test_old"
classes = ["false", "true"]
tfrecord_file = "data_test_old.tfrecord"
jpgs = []

def convert_dataset(jpgs, tfrecord_file):
    with tf.compat.v1.python_io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        for jpg in jpgs:
            img_ir = cv2.imread(jpg)
            img_rd = cv2.imread(jpg.replace('IR', 'RD'))
            print(jpg, classes.index(os.path.basename(os.path.dirname(os.path.dirname(jpg)))))
            try:
                example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(
                    feature={"label": tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[classes.index(os.path.basename(os.path.dirname(os.path.dirname(jpg))))])),
                             "image_ir": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[img_ir.tostring()])),
                             "image_rd": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[img_rd.tostring()])),
                             "filename": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[jpg.encode()]))}))
            except AttributeError:
                continue
            tfrecord_writer.write(example.SerializeToString())

for classe in classes:
    for image in os.listdir(os.path.join(dataset_dir, classe, 'IR')):
        jpgs.append(os.path.join(dataset_dir, classe, 'IR', image))
random.shuffle(jpgs)
print(jpgs)
convert_dataset(jpgs, tfrecord_file)
print("num:", len(jpgs))
