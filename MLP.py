""" Multilayer Perceptron.
"""


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

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

mnist_train_data=mnist_train_data.flatten()
mnist_train_data=mnist_train_data.reshape(60000,45*45)
mnist_test_data=mnist_test_data.flatten()
mnist_test_data=mnist_test_data.reshape(10000,45*45)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
training_steps = 10000
batch_size = 100
display_step = 100

# Network Parameters
n_hidden_1 = 516 # 1st layer number of neurons
n_hidden_2 = 516 # 2nd layer number of neurons
n_input = 45*45 # MNIST data input (img shape: 45*45)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
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

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(Y, dtype=tf.int32)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, dtype=tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

step_num=[]
loss_value=[]
accu_value=[]

with tf.Session() as sess:
    sess.run(init)

    for step in range(1, training_steps + 1):
        batch_x, batch_y = next_batch(mnist_train_data, mnist_train_label, step, batch_size)
        # Reshape data to get 45 seq of 45 elements
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

    # Calculate accuracy for mnist test images
    test_data = mnist_test_data
    test_label = mnist_test_label
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

# Plot the loss and accuracy of MLP
fig=plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.plot(step_num, loss_value, 'r')
ax2.plot(step_num, accu_value, 'b')
ax1.set_xlabel("step")
ax1.set_ylabel("loss")
ax2.set_xlabel("step")
ax2.set_ylabel("accuracy")

plt.show()