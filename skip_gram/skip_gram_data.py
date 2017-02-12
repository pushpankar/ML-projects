from __future__ import print_function
import collections
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

# Download Dataset

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    "Download file if not present"
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url+filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified {}'.format(filename))
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify')
    return filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    """Extract the first file enclosed in zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


words = read_data(filename)
print('Data size is {}'.format(len(words)))

# Remove words to keep vocabulary size reasonable
# Remove rare words with 'UNK' token
vocabulary_size = 50000


def build_dataset(words):
    # Create a list of tuples where each tuple contains word and count
    count = [('UNK', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    # Change the list of tuples to a dictionary and give each word a number
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)
    count[0] = ('UNK', unk_count)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])

# Create training batch for skip gram
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
