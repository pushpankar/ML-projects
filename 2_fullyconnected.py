from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
sess = tf.InteractiveSession()

pickle_file = "../tensorflow/tensorflow/examples/udacity/notMNIST.pickle"

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']

    valid_dataset = save['valid_dataset']
    valid_labels  = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']

    del save #hint to help gc free up memory

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
#reformat the data that is more adaptive to the models
image_size = 28
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size*image_size)).astype(np.float32)
    #one hot key encoding
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset,  test_labels  = reformat(test_dataset,  test_labels)

print('Training set' , train_dataset.shape, train_labels.shape)
print('Validation set' , valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# now create a graph using tensorflow
x = tf.placeholder(tf.float32, shape=[None,784])
y_ = tf.placeholder(tf.float32, shape=[None,10])

W1 = tf.Variable(tf.truncated_normal([image_size*image_size,1024],stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,shape=[1024]))
y1 = tf.matmul(x,W1) + b1
h_fc1 = tf.nn.relu(y1)

W2 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[512]))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W2) + b2)

W = tf.Variable(tf.truncated_normal([512,  num_labels], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[num_labels]))
y = tf.nn.softmax(tf.matmul(h_fc2,W) + b)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess.run(tf.global_variables_initializer())


batch_size = 128

for i in range(6001):
    offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {x:batch_data, y_:batch_labels}
    train_step.run(feed_dict)

    if i%500 == 0:
        train_accuracy = accuracy.eval(feed_dict)
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print("validation accuracy %g"%accuracy.eval(feed_dict={x:valid_dataset, y_:valid_labels}))

print("test accuracy %g"%accuracy.eval(feed_dict={x:test_dataset,y_:test_labels}))
        
    
