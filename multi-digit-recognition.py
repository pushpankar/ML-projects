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
num_labels = 11

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2) == np.argmax(labels, 2))/ predictions.shape[0])/5

train_dataset, train_labels = reformat(mnist.train.images, mnist.train.labels)
valid_dataset, valid_labels = reformat(mnist.validation.images, mnist.validation.labels)
test_dataset, test_labels = reformat(mnist.test.images, mnist.test.labels)

#synthesize images: merge 5 image to create 1 image
def merge_images(dataset,labels,num_images):
    total_images = dataset.shape[0]
    random_pos = np.random.randint(total_images,size=num_images)
    xs = dataset[random_pos]
    ys = labels[random_pos]
    ys = np.append(ys, np.zeros((num_images,1)), axis=1)

    #insert random number of blank spaces 
    replace_count = np.random.randint(3)
    for _ in range(replace_count):
        if np.random.randint(10) < 3:
            blank_pos = np.random.randint(0,4)
            xs[blank_pos] = np.random.randint(0,255,size=(28,28,1)).astype(np.float32)
            ys = np.delete(ys, (blank_pos), axis=0)
            ys = np.append(ys,np.zeros((1,11)),axis=0)
            ys[num_images-1][10] = 1
        
    concatanated_image = np.concatenate(xs, axis=1)
    return concatanated_image, ys

#create a set of merged images
def get_samples(num_samples):
    sample = []
    labels = []
    for i in xrange(num_samples):
        data, label = merge_images(train_dataset,train_labels,5)
        sample.append(data)
        labels.append(label)
    return np.array(sample), np.array(labels)

def create_validation_and_test():
    sample_valid_list = []
    labels_valid_list = []
    sample_test_list = []
    labels_test_list = []
    for i in xrange(250):
        merged_data_valid, merged_label_valid = merge_images(valid_dataset, valid_labels,5)
        merged_data_test, merged_label_test  = merge_images(test_dataset, test_labels,5)
        sample_valid_list.append(merged_data_valid)
        labels_valid_list.append(merged_label_valid)
        sample_test_list.append(merged_data_test)
        labels_test_list.append(merged_label_test)

    new_valid_dataset = np.array(sample_valid_list)
    new_valid_labels = np.array(labels_valid_list)
    new_test_dataset = np.array(sample_test_list)
    new_test_labels = np.array(labels_test_list)
    return (new_valid_dataset, new_valid_labels, new_test_dataset, new_test_labels)
        
valid_dataset, valid_labels, test_dataset, test_labels = create_validation_and_test()
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


#check dimensions
sample = get_samples(50)
print(sample[0].shape, sample[1].shape)

batch_size = 50
patch_size = 5
depth = 128
num_hidden = 512
image_width = 140
image_height = 28


#build a graph
graph = tf.Graph()
with graph.as_default():
    #placeholder for input data and labels
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_height, image_width, num_channels))
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
    layer4_W = weight_variable([9216, num_hidden])
    layer4_bias = bias_variable([num_hidden])

    layer5_Ws = [weight_variable([num_hidden, num_labels]) for _ in xrange(5)]
    layer5_biases = [bias_variable([num_labels]) for _ in xrange(5)]


    #Model
    def model(data):
        conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_W, [1,2,2,1], padding='SAME') + layer1_bias)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, layer2_W, [1,2,2,1], padding='SAME') + layer2_bias)
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, layer3_W, [1,2,2,1], padding='SAME') + layer3_bias)
        shape = conv3.get_shape().as_list()
        reshaped = tf.reshape(conv3, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden4 = tf.nn.relu(tf.matmul(reshaped, layer4_W) + layer4_bias)
        return [tf.matmul(hidden4, layer5_W) + layer5_bias for layer5_W, layer5_bias in zip(layer5_Ws, layer5_biases)]

    #A list of logits
    logits = model(tf_train_dataset)
    loss_per_digit = [tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits[i],tf_train_labels[:,i,:])) for i in xrange(5)]
    loss = tf.add_n(loss_per_digit)

    #optimizer
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    train_prediction = tf.transpose(tf.nn.softmax(logits), [1,0,2])
    valid_prediction = tf.transpose(tf.nn.softmax(model(tf_valid_dataset)), [1,0,2])
    test_prediction  = tf.transpose(tf.nn.softmax(model(tf_test_dataset)), [1,0,2])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(9001):
        batch = get_samples(batch_size)
        feed_dict = { tf_train_dataset: batch[0],
                      tf_train_labels: batch[1]}
        _,l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 100 == 0:
            print('Minibatch accuracy at step {} : {}'.format(step, accuracy(predictions,  batch[1])))
            #print('Minibatch loss at step{}: {}'.format(step,l))
            print('Validation accuracy at step {}: {}'.format(step, accuracy(valid_prediction.eval(), valid_labels))) 

    print('Test accuracy is {}'.format(accuracy(test_prediction.eval(), test_labels)))
