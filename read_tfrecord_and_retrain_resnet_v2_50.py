import tensorflow as tf
import tensorflow.contrib.slim as slim
from resnet_utils import resnet_arg_scope
from resnet_v2_50 import resnet_v2_50

checkpoint_file = "resnet_v2_50.ckpt"
tfrecord_file = "data_train.tfrecord"

image_pixels = 224
classes = 2
epochs = 200
train_size = 7678
batch_size = 20

def read_and_decode(serialized_example):
    features = tf.compat.v1.parse_single_example(serialized_example, features={"label":tf.compat.v1.FixedLenFeature([], tf.compat.v1.int64),
                                                                               "image_ir":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string),
                                                                               "image_rd":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img_ir = tf.compat.v1.decode_raw(features["image_ir"], tf.compat.v1.uint8)
    img_ir = tf.compat.v1.reshape(img_ir, [image_pixels, image_pixels, 3])

    img_rd = tf.compat.v1.decode_raw(features["image_rd"], tf.compat.v1.uint8)
    img_rd = tf.compat.v1.reshape(img_rd, [image_pixels, image_pixels, 3])
    
    label = tf.compat.v1.cast(features["label"], tf.compat.v1.int64)
    return img_ir, img_rd, label

images_ir = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input_ir")
images_rd = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, image_pixels, image_pixels, 3], name="input/x_input_rd")
labels = tf.compat.v1.placeholder(tf.compat.v1.int64, [None], name="input/y_input")

with slim.arg_scope(resnet_arg_scope()):
    logits, end_points = resnet_v2_50(images_ir, images_rd, num_classes=classes, is_training=True)

exclude = ["resnet_v2_50/logits"]
variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

learning_rate = tf.compat.v1.Variable(initial_value=0.0001, trainable=False, name="learning_rate", dtype=tf.compat.v1.float32)
update_learning_rate = tf.compat.v1.assign(learning_rate, learning_rate*0.8)
one_hot_labels = slim.one_hot_encoding(labels, classes)
loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
total_loss = tf.compat.v1.losses.get_total_loss()
update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
with tf.compat.v1.control_dependencies(update_ops):
    train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)
correct_prediction = tf.compat.v1.equal(labels, tf.compat.v1.argmax(end_points['predictions'], 1), name="correct_prediction")
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(correct_prediction, tf.compat.v1.float32), name="accuracy")

with tf.compat.v1.Session() as sess:
    tf.compat.v1.global_variables_initializer().run()
    # tf.compat.v1.train.Saver(variables_to_restore).restore(sess, checkpoint_file)
    # ckpt = tf.compat.v1.train.get_checkpoint_state("ckpt")
    # if ckpt:
    #     print(ckpt.model_checkpoint_path)
    #     tf.compat.v1.train.Saver(var_list=slim.get_variables_to_restore()).restore(sess, ckpt.model_checkpoint_path)
    # else:
    #     raise ValueError('The ckpt file is None.')
    tf.compat.v1.summary.FileWriter("logs/", sess.graph)
    dataset_train = tf.compat.v1.data.TFRecordDataset([tfrecord_file])
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)

    for epoch in range(epochs):
        if epoch != 0 and epoch % 10 == 0:
            sess.run(update_learning_rate)
        print("learning_rate:", sess.run(learning_rate))
        for step in range(int(train_size/batch_size)):
            img_train_ir, img_train_rd, label_train = sess.run(next_element_train)
            _, _total_loss, _accuracy = sess.run([train_step, total_loss, accuracy], feed_dict={images_ir:img_train_ir, images_rd:img_train_rd, labels:label_train})
            if step % 10 == 0:
                print("step:", step / 10, " total_loss:", _total_loss, " accuracy:", _accuracy)
        tf.compat.v1.train.Saver().save(sess, "ckpts/model.ckpt", global_step=epoch)
        print("save ckpt:", epoch)
