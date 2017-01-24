from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = "../tensorflow/tensorflow/examples/udacity/notMNIST.pickle"

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)


image_size = 28
num_labels = 10
num_channels = 1

def reformat(dataset, labels):
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()
with graph.as_default():
    #input data
    tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size,
                           num_channels))
    tf_train_labels = tf.placeholder(
        tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    #variables
    W_L1 = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, num_channels, depth], stddev=0.1))
    b_L1 = tf.Variable(tf.zeros([depth]))

    #Layer 2
    W_L2 = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
    b_L2 = tf.Variable(tf.constant(1.0, shape=[depth]))

    #Layer 3
    W_L3 = tf.Variable(tf.truncated_normal(
      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
    b_L3 = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    #Layer 4
    W_L4 = tf.Variable(tf.truncated_normal(
        [num_hidden, num_labels], stddev=0.1))
    b_L4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))


    #model
    def model(data):
        conv1 = tf.nn.conv2d(data,W_L1, [1,2,2,1], padding='SAME')
        hidden1 = tf.nn.relu(conv1 + b_L1)
        conv2 = tf.nn.conv2d(conv1, W_L2, [1,2,2,1], padding='SAME')
        hidden2 = tf.nn.relu(conv2 + b_L2)
        shape = hidden2.get_shape().as_list()
        reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden3 = tf.nn.relu(tf.matmul(reshape, W_L3) + b_L3)
        return tf.matmul(hidden3, W_L4) + b_L4

    #Training computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    #predections
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 10001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset:batch_data,
                     tf_train_labels:batch_labels}
        
        _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict = feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(
                valid_prediction.eval(), valid_labels))
            
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
                
