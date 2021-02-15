import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
from scipy.misc import imread
from caffe_classes import class_names

# TODO: Load traffic signs data.
training_data = './train.p'
nb_classes = 43
EPOCHS = 1
BATCH_SIZE = 128
meanweight = 0.0
standardDev = 0.1
with open(training_data, mode='rb') as f:
    data = pickle.load(f)
# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)
# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
keep_prob = tf.placeholder(
    tf.float32)  # defining the dropout probability after fully connected layer in the architecture
one_hot_labels = tf.one_hot(labels, nb_classes)
resized_features = tf.image.resize_images(features, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_features, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
logits = tf.add(tf.matmul(fc7, tf.Variable(tf.random_normal(shape=shape, mean=meanweight, stddev=standardDev))),
                tf.Variable(tf.random_normal([43])))
# TODO: Add the final layer for traffic sign classification.

probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
loss_operation = tf.reduce_mean(probs)  # define loss function
optimizer = tf.train.AdamOptimizer(learning_rate=0.0009)
training_operation = optimizer.minimize(loss_operation)

pred = tf.arg_max(logits, 1)
# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(pred, labels)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    # num_examples = len(X_data)
    num_examples = X_data.shape[0]
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, X_data.shape[0], BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation],
                                  feed_dict={features: batch_x, labels: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))  # getting the total loss to plot a graph later
    return total_accuracy / num_examples, total_loss / num_examples


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
#
#     print("Training...")
#     print()
#     loss_Acc = []
#     for i in range(EPOCHS):
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y, keep_prob: 0.5})
#
#         validation_accuracy, loss_acc = evaluate(X_valid, y_valid)
#
#         print("EPOCH {} ...".format(i + 1))
#         loss_Acc.append(loss_acc)
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#     plt.plot(range(0, EPOCHS), loss_Acc)
#     plt.ylabel('loss')
#     plt.xlabel('Epochs')
#     plt.grid(True)
#     plt.show()
#     saver.save(sess, './alexnettrain')
#     print("Model saved")


testing_file = '../CNN/traffic-signs-data/test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_test, y_test = test['features'], test['labels']

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy[0]))

# # Check the traffic signs again
# import os
# import matplotlib.image as mpimg
# import cv2
#
# my_images = []
#
# for i, img in enumerate(os.listdir('./traffic-signs-real/new/')):
#     image = cv2.imread('./traffic-signs-real/new/' + img)
#     my_images.append(image)
#     plt.figure()
#     plt.xlabel(img)
#     plt.imshow(image)
#     plt.show()
#
# my_images = np.asarray(my_images)
#
# my_labels = [35, 29, 17, 27, 31]
# # Check Test Accuracy
#
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     output_accuracy = evaluate(my_images, my_labels)
#     print("Test Accuracy = {:.3f}".format(output_accuracy[0]))
