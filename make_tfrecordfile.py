import os
import random
from PIL import Image
import tensorflow as tf


dataset_dir = "data/test"
classes = ["false", "true"]
tfrecord_file = "data_test.tfrecord"
jpgs = []

def convert_dataset(jpgs, tfrecord_file):
    with tf.compat.v1.python_io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        for jpg in jpgs:
            img_ir = Image.open(jpg)
            img_rd = Image.open(jpg.replace('IR', 'RD'))
            example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(
                feature={"label": tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[classes.index(os.path.basename(os.path.dirname(os.path.dirname(jpg))))])),
                         "image_ir": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[img_ir.tobytes()])),
                         "image_rd": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[img_rd.tobytes()])),
                         "filename": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[jpg.encode()]))}))
            print(jpg, classes.index(os.path.basename(os.path.dirname(os.path.dirname(jpg)))))
            tfrecord_writer.write(example.SerializeToString())

for classe in classes:
    for image in os.listdir(os.path.join(dataset_dir, classe, 'IR')):
        jpgs.append(os.path.join(dataset_dir, classe, 'IR', image))
random.shuffle(jpgs)
print(jpgs)
convert_dataset(jpgs, tfrecord_file)
print("num:", len(jpgs))