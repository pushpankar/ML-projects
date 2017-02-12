from skip_gram_data import maybe_download, read_data, build_dataset, generate_batch
import math
import random
import numpy as np
import tensorflow as tf

vocabulary_size = 50000

filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)
print('data size is {}'.format(len(words)))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(
        batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\n with num_skips = {} and skip_window = {}'.format(
        num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


# Train skip_gram_model
batch_size = 128
embedding_size = 128  # Dimension of embedding vector
skip_window = 1  # Number of words to look in left and right
num_skips = 2  # Number of times to reuse an input to generate a label
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    # Input data
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Variables
    embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_bias = tf.Variable(tf.zeros([vocabulary_size]))

    # Model
    embed = tf.nn.embedding_lookup(embedding, train_dataset)
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_bias, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    # Optimizer
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embeddings = embedding / norm
    valid_embedding = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embeddings))

num_steps = 100001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_label = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {
            train_dataset: batch_data,
            train_labels: batch_label
        }
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
            average_loss = average_loss / 2000
            # The average_loss is an estimate of the loss over the last 2000 batches
            print('average_loss at step {} is {}'.format(step, average_loss))
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbours
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
                final_embeddings = normalized_embeddings.eval()
