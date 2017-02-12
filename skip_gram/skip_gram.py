from skip_gram_data import maybe_download, read_data, build_dataset, generate_batch

vocabulary_size = 50000

filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)
print('data size is {}'.format(len(words)))

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\n with num_skips = {} and skip_window = {}'.format(num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
