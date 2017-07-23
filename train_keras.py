import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time

from include.data import get_data_set
# from include.model import model
from include.model_TA import model

from utils import progress_bar


x, y, output, global_step, y_pred_cls, keep_prob = model()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 20000
_EPOCH = 300
# _SAVE_PATH = "./tensorboard/cifar-10/"
_SAVE_PATH = "./tensorboard1/cifar-10/"

train_x, train_y, train_l = get_data_set(cifar=10)
test_x, test_y, test_l = get_data_set("test", cifar=10)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
steps_per_epoch = len(train_x) / _BATCH_SIZE
boundaries = [steps_per_epoch * _epoch for _epoch in [10, 40]]
values = [0.1, 0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
weight_decay = 0.0001
# optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss + l2 * weight_decay, global_step=global_step)
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, name='Momentum', use_nesterov=True).minimize(loss + l2 * weight_decay, global_step=global_step)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy/train", accuracy)


merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_epoch):
    '''
        Train CNN
    '''
    global train_x
    global train_y

    epoch_size = len(train_x)
    for i in range(num_epoch):
        print ('Epoch: %d' % i)
        randidx = np.arange(epoch_size)
        np.random.shuffle(randidx)
        print (epoch_size)

        train_x = train_x[randidx]
        train_y = train_y[randidx]

        if (epoch_size % _BATCH_SIZE == 0):
            num_iterations = epoch_size / _BATCH_SIZE
        else:
            num_iterations = int(epoch_size / _BATCH_SIZE) + 1

        train_loss = 0
        for j in range(num_iterations):
            if (j < num_iterations - 1):
                batch_xs = train_x[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
                batch_ys = train_y[j * _BATCH_SIZE:(j + 1) * _BATCH_SIZE]
            else:
                batch_xs = train_x[j * _BATCH_SIZE:epoch_size]
                batch_ys = train_y[j * _BATCH_SIZE:epoch_size]

            start_time = time()
            i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            duration = time() - start_time

            if (i_global % 10 == 0) or (j == num_iterations - 1):
                _loss, batch_acc, _learning_rate = sess.run([loss, accuracy, learning_rate], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
                # msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
                # print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))
                train_loss = train_loss + _loss
                progress_bar(j, num_iterations, 'Loss: %.3f | Acc: %.3f%% '
                             % (train_loss / (j + 1), batch_acc))

            # if (i_global % 100 == 0) or (i == num_iterations - 1):
            if (j == num_iterations - 1):
                data_merged, global_1 = sess.run([merged, global_step], feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
                acc = predict_test()

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
                ])
                train_writer.add_summary(data_merged, global_1)
                train_writer.add_summary(summary, global_1)

                saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
                print("Saved checkpoint.")


def predict_test(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))
        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean() * 100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc


if _ITERATION != 0:
    # train(_ITERATION)
    train(_EPOCH)


sess.close()
