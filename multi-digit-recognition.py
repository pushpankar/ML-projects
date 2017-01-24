import math
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(1.0, shape=shape))

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,2,2,1], padding='SAME')

image_size = 28
num_channels = 1
num_labels = 10

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(mnist.train.images, mnist.train.labels)
valid_dataset, valid_labels = reformat(mnist.validation.images, mnist.validation.labels)
test_dataset, test_labels = reformat(mnist.test.images, mnist.test.labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)



def merge_images(dataset,labels,num_images):
    total_images = dataset.shape[0]
    random_pos = np.random.randint(total_images,size=num_images)
    xs = dataset[random_pos]
    ys = labels[random_pos]
    concatanated_image = np.concatenate(xs, axis=1)
    return concatanated_image, ys

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


def get_samples(num_samples):
    sample = []
    labels = []
    for i in xrange(num_samples):
        data, label = merge_images(train_dataset,train_labels,5)
        sample.append(data)
        labels.append(label)
    return np.array(sample), np.array(labels)

print(merge_images(train_dataset,train_labels,5)[1].shape)
print(get_samples(70)[1].shape)
print(get_samples(70)[0].shape)


batch_size = 50
patch_size = 5
depth = 64
num_hidden = 128
image_width = 140
image_height = 28
#lets build a graph
graph = tf.Graph()
with graph.as_default():
    #placeholder for input data and labels

    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, num_channels))
    #We have 5 lists of a batch of labels
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, 5, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    #create variables
    #Convolutions
    #layer1
    layer1_W = weight_variable([patch_size, patch_size, num_channels, depth])
    layer1_bias = bias_variable([depth])

    #layer2
    layer2_W = weight_variable([patch_size, patch_size, depth, depth])
    layer2_bias = bias_variable([depth])

    #layer3
    layer3_W = weight_variable([patch_size, patch_size, depth, depth])
    layer3_bias = bias_variable([depth])

    #Fully connected layers
    layer4_W = weight_variable([4608, num_hidden])
    layer4_bias = bias_variable([num_hidden])

    layer5_Ws = [weight_variable([num_hidden, num_labels]) for _ in xrange(5)]
    layer5_biases = [bias_variable([num_labels]) for _ in xrange(5)]


    #Model
    def model(data):
        conv1 = tf.nn.relu(tf.nn.conv2d(tf_train_dataset, layer1_W, [1,2,2,1], padding='SAME') + layer1_bias)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, layer2_W, [1,2,2,1], padding='SAME') + layer2_bias)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, layer3_W, [1,2,2,1], padding='SAME') + layer3_bias)
        shape = conv3.get_shape().as_list()
        reshaped = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden4 = tf.nn.relu(tf.matmul(reshaped, layer4_W) + layer4_bias)
        return [tf.matmul(hidden4, layer5_W) + layer5_bias for layer5_W, layer5_bias in zip(layer5_Ws, layer5_biases)]

    logits = model(tf_train_dataset)
    print(logits[1])
    loss_per_digit = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[i],tf_train_labels[:,i,:])) for i in range(5)]
    loss = sum(loss_per_digit)

    #optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction  = tf.nn.softmax(model(tf_test_dataset))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(5000):
        batch = get_samples(batch_size)
        feed_dict = { tf_train_dataset: batch[0],
                      tf_train_labels: batch[1]}
        _,l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step{}: {}'.format(step,l))
    
