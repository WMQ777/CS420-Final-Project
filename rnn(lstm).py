""" Recurrent Neural Network.
    LSTM
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
mnist_train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
mnist_train_label = np.fromfile("mnist_train_label",dtype=np.uint8)
mnist_test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
mnist_test_label = np.fromfile("mnist_test_label",dtype=np.uint8)


mnist_train_data = mnist_train_data.reshape(60000,45,45)
mnist_train_data = mnist_train_data.astype(np.float32)
mnist_test_data = mnist_test_data.reshape(10000,45,45)
mnist_test_data = mnist_test_data.astype(np.float32)
mnist_train_label = mnist_train_label.astype(np.int32)
mnist_test_label = mnist_test_label.astype(np.int32)
'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 45*45px, we will then
handle 45 sequences of 45 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 100
display_step = 100

# Network Parameters
num_input = 45 # MNIST data input (img shape: 45*45)
timesteps = 45 # timesteps
num_hidden = 512 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def next_batch(x, y, batch, batch_size):
    num=int(x.shape[0]/batch_size)
    batch=batch%num
    if batch == 0:
        index = [i for i in range(0, x.shape[0])]
        np.random.shuffle(index)
        x=x[index]
        y=y[index]
    start = batch_size*batch
    end = batch_size * (batch+1)
    return x[start:end], y[start:end]

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(Y, dtype=tf.int32)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, dtype=tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
step_num=[]
loss_value=[]
accu_value=[]

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = next_batch(mnist_train_data, mnist_train_label, step, batch_size)
        # Reshape data to get 45 seq of 45 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            step_num.append(step)
            loss_value.append(loss)
            accu_value.append(acc)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images

    test_data = mnist_test_data
    test_label = mnist_test_label
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

#Plot the loss and accuracy of RNN(LSTM)
plt.figure()
plt.plot(step_num,loss_value,'r',label="loss")
plt.plot(step_num,accu_value,'b',label="accuracy")
plt.legend()
plt.xlabel("step")
plt.title("RNN (LSTM)")
plt.show()