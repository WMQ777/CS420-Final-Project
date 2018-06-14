""" Convolutional Neural Network.
    AlexNet model
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

# Training Parameters
learning_rate = 0.001
num_steps = 5000
batch_size = 128

# Network Parameters
num_input = 45*45 # MNIST data input (img shape: 45*45)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit

# Load MNIST
mnist_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
mnist_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
mnist_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
mnist_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)
print(mnist_train_data.shape)

data_num = 60000 #The number of figures
fig_w = 45       #width of each figure
mnist_train_data = mnist_train_data.reshape(60000,45,45)
mnist_train_data = mnist_train_data.astype(np.float32)
mnist_test_data = mnist_test_data.reshape(10000,45,45)
mnist_test_data = mnist_test_data.astype(np.float32)
print(mnist_train_data.shape, type(mnist_train_data), mnist_train_data.dtype)
print(mnist_test_data.shape, type(mnist_test_data), mnist_test_data.dtype)

mnist_train_label = mnist_train_label.astype(np.int32)
mnist_test_label = mnist_test_label.astype(np.int32)
print(mnist_train_label.shape, type(mnist_train_label), mnist_train_label.dtype)
print(mnist_test_label.shape, type(mnist_test_label), mnist_test_label.dtype)

# 定义函数print_activations来显示网络每一层结构，展示每一个卷积层或池化层输出tensor的尺寸
def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())

# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 45, 45, 1])

        # Convolution Layer with 64 filters and a kernel size of 11
        conv1 = tf.layers.conv2d(x, 64, 3, padding='SAME', activation=tf.nn.relu)
        print("Convolution 1:")
        print_activations(conv1)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='SAME')
        print("Max pooling 1:")
        print_activations(conv1)

        # Convolution Layer with 192 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 128, 3, padding='SAME', activation=tf.nn.relu)
        print("Convolution 2:")
        print_activations(conv2)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME')
        print("Max pooling 2:")
        print_activations(conv2)

        # Convolution Layer with 384 filters and a kernel size of 3
        conv3 = tf.layers.conv2d(conv2, 256, 3, padding='SAME', activation=tf.nn.relu)
        print("Convolution 3:")
        print_activations(conv3)

        # Convolution Layer with 384 filters and a kernel size of 3
        conv4 = tf.layers.conv2d(conv3, 256, 3, padding='SAME', activation=tf.nn.relu)
        print("Convolution 4:")
        print_activations(conv4)

        # Convolution Layer with 384 filters and a kernel size of 3
        conv5 = tf.layers.conv2d(conv4, 128, 3, padding='SAME', activation=tf.nn.relu)
        print("Convolution 5:")
        print_activations(conv5)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv5 = tf.layers.max_pooling2d(conv5, 2, 2, padding='SAME')
        print("Max pooling 5:")
        print_activations(conv5)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv5)
        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 4096, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        print("Fully connected 1:")
        print_activations(fc1)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc2 = tf.contrib.layers.flatten(fc1)
        # Fully connected layer (in tf contrib folder for now)
        fc2 = tf.layers.dense(fc2, 4096, activation=tf.nn.relu)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc2 = tf.layers.dropout(fc2, rate=dropout, training=is_training)
        print("Fully connected 2:")
        print_activations(fc2)

        # Output layer, class prediction
        out = tf.layers.dense(fc2, num_classes)
        print("Out layer:")
        print_activations(out)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    print("Begin optimizer!")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())
    print("End optimizer!")

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    print("loss_op:",loss_op, "acc_op:",acc_op)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})
    print("estim_specs:",estim_specs)

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist_train_data}, y=mnist_train_label,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
print("Begin training the model!")
model.train(input_fn, steps=num_steps)
print("End training the model!")

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist_test_data}, y=mnist_test_label,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
