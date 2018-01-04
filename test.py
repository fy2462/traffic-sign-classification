# Load pickled data
import pickle
from sklearn.model_selection import train_test_split

# TODO: Fill this in based on where you saved the training and testing data

training_file = '/Users/billzito/Documents/Udacity/sdc/p2-traffic-signs/CarND-Traffic-Signs/traffic-signs-data/train.p'
testing_file = '/Users/billzito/Documents/Udacity/sdc/p2-traffic-signs/CarND-Traffic-Signs/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, X_validate, y_train, y_validate = train_test_split(train['features'], train['labels'], test_size=0.10, random_state=0)
# X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("validation set: {} samples".format(len(X_validate)))
print("Test Set:       {} samples".format(len(X_test)))


### Replace each question mark with the appropriate value.

n_train = X_train.shape[0]
n_validate = X_validate.shape[0]
n_test =  X_test.shape[0]
image_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
n_classes = y_train.max() + 1

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validate)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt
from random import randint

for num in range(10):
    rand = randint(0, 12630)
    plt.figure()
    plt.imshow(X_test[rand])

plt.show()

import cv2
import numpy as np
print("Image Shape: {}".format(X_train[0].shape) + '\n')
orig = X_train[0].shape
test = np.reshape(cv2.cvtColor(X_train[0], cv2.COLOR_BGR2GRAY), (orig[0], orig[1], 1))
print("test shape: {}".format(test.shape) + '\n')
# plt.imshow(test, cmap='gray')
X_train_gray = [np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (orig[0], orig[1], 1)) for img in X_train]
X_validate_gray = [np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (orig[0], orig[1], 1)) for img in X_validate]
X_test_gray = [np.reshape(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (orig[0], orig[1], 1)) for img in X_test]

print("after grayed Shape: {}".format(X_train_gray[0].shape) + '\n')
# print(len(X_train_gray), len(X_validate_gray), len(X_test_gray))

from sklearn.utils import shuffle

X_train_gray, y_train = shuffle(X_train_gray, y_train)

import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten

# graph = tf.Graph()
# with graph.as_default():

patch_size = 5
num_channels = 1  # grayscale
depth_1 = 6
depth_2 = 16
full_1_output = 120
full_2_output = 84
full_3_output = 43
dropout_keepers = .8


def LeNet(x):
    # Hyperparameters
    mu = 0

    # why use stddev 0.1?
    sigma = 0.1

    weights = {
        'layer_1': tf.Variable(
            tf.truncated_normal([patch_size, patch_size, num_channels, depth_1], mean=mu, stddev=sigma)),
        'layer_2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_1, depth_2], mean=mu, stddev=sigma)),
        'layer_3': tf.Variable(tf.truncated_normal([400, full_1_output], mean=mu, stddev=sigma)),
        'layer_4': tf.Variable(tf.truncated_normal([full_1_output, full_2_output], mean=mu, stddev=sigma)),
        'layer_5': tf.Variable(tf.truncated_normal([full_2_output, full_3_output], mean=mu, stddev=sigma))
    }

    biases = {
        'layer_1': tf.Variable(tf.zeros([depth_1])),
        'layer_2': tf.Variable(tf.zeros(shape=[depth_2])),
        'layer_3': tf.Variable(tf.zeros(shape=[full_1_output])),
        'layer_4': tf.Variable(tf.zeros(shape=[full_2_output])),
        'layer_5': tf.Variable(tf.zeros(shape=[full_3_output]))
    }

    # getting rid of one col and one row on each side with valid will only move to 30, 30? I guess not because
    # deleting rows also removes columns--> 4 lost
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv = tf.nn.conv2d(x, weights['layer_1'], [1, 1, 1, 1], padding='VALID')
    #   print('conv is', conv)

    # TODO: Activation.
    hidden_1 = tf.nn.dropout(tf.nn.relu(conv + biases['layer_1']), dropout_keepers)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    max_1 = tf.nn.max_pool(hidden_1, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    #   print('max 1', max_1)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv_2 = tf.nn.conv2d(max_1, weights['layer_2'], [1, 1, 1, 1], padding="VALID")
    #   print('conv_2', conv_2)

    # TODO: Activation.
    hidden_2 = tf.nn.dropout(tf.nn.relu(conv_2 + biases['layer_2']), dropout_keepers)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    max_2 = tf.nn.max_pool(hidden_2, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    #   print('max 2', max_2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    # multiply all vals times each other-- reshape to -1?
    #   max_shape = tf.size(max_2)
    #   print(max_shape)
    #   flat = tf.reshape(max_2, [-1, 5 * 5 * 16])
    #   print('flat', flat)
    flat = flatten(max_2)

    # tried to do these together
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # TODO: Activation.
    full_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(flat, weights['layer_3']) + biases['layer_3']), dropout_keepers)
    #   print('full_1 is', full_1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    full_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(full_1, weights['layer_4']) + biases['layer_4']), dropout_keepers)
    #   print('full_2 is', full_2)

    # TODO: Activation.

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.matmul(full_2, weights['layer_5']) + biases['layer_5']
    #     print('logits are', logits)

    return logits, weights

    # l, w = LeNet(X_train)
    # print('weights are', w)


### Train your model here.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
last_validation_score = 0

EPOCHS = 30

rate = 0.001
# beta = 0.001

logits, weights = LeNet(x)
# l2 = tf.nn.l2_loss(weights['layer_1']) + tf.nn.l2_loss(weights['layer_2']) + tf.nn.l2_loss(weights['layer_3']) + tf.nn.l2_loss(weights['layer_4']) + tf.nn.l2_loss(weights['layer_5'])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
# cross_entropy + beta * l2
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_gray)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_gray, y_train = shuffle(X_train_gray, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_gray[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validate_gray, y_validate)
        #         if validation_accuracy < last_validation_score
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    test_accuracy = evaluate(X_test_gray, y_test)
    print('you know who i am', test_accuracy)

    try:
        saver
    except NameError:
        saver = tf.train.Saver()
    saver.save(sess, 'lenet')
    print("Model saved")

    ### Load the images and plot them
    ### convert images to correct shape and grayscale
    ### predict and print prediction
    import os
    import matplotlib.image as mpimg

    dir = 'wild-signs/'
    files = os.listdir(dir)

    sess = tf.Session()
    saver.restore(sess, './lenet')

    for count, filename in enumerate(files):
        plt.figure()
        img = mpimg.imread(dir + filename)
        plt.imshow(img)

        resized = cv2.resize(img, (32, 32))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        plt.figure()
        plt.imshow(gray, cmap='gray')
        gray = np.reshape(gray, (1, 32, 32, 1)).astype(np.int32)
        scores = logits.eval(feed_dict={x: gray}, session=sess)
        #     softmax = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
        prediction = tf.argmax(scores, 1)
        print('prediction', filename, prediction.eval(session=sess))

    ### Visualize the softmax probabilities here.
    ### Feel free to use as many code cells as needed.
    for count, filename in enumerate(files):
        img = mpimg.imread(dir + filename)
        resized = cv2.resize(img, (32, 32))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = np.reshape(gray, (1, 32, 32, 1)).astype(np.int32)

        scores = logits.eval(feed_dict={x: gray}, session=sess)
        softmax = tf.nn.softmax(scores)
        #     print('softmax', softmax)
        top_3 = sess.run(tf.nn.top_k(softmax, k=3))
        print('prediction', filename, top_3)