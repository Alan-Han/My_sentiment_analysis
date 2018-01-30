import numpy as np
import pickle
from collections import Counter
from string import punctuation

def get_data():
    with open('inputs/reviews.txt', 'r') as f:
        reviews = f.read()
    with open('inputs/labels.txt', 'r') as f:
        labels = f.read()
    return reviews, labels

def preprocess_data(data):
    """
    Get rid of periods and split the text into each review using \n,then split it into individual words
    """
    all_text = ''.join([c for c in data if c not in punctuation])
    reviews = all_text.split('\n')
    all_text = ' '.join(reviews)
    words = all_text.split()
    return reviews, words

def encode_data(data, reviews):
    """
    encode the words with integers
    """
    counts = Counter(data)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}  # pad input vec with 0, so start with 1
    n_words = len(vocab_to_int)
    reviews_ints = []

    for each in reviews:
        reviews_ints.append([vocab_to_int[word] for word in each.split()])
    return reviews_ints, n_words

def encode_label(labels):
    labels = labels.split('\n')
    labels = np.array([1 if each == 'positive' else 0 for each in labels])
    return labels



def preprocess(seq_len=200, split_frac=0.8):
    reviews, labels = get_data()

    reviews, words = preprocess_data(reviews)

    reviews_ints, n_words = encode_data(words, reviews)

    labels = encode_label(labels)

    pickle.dump(n_words, open('preprocess_data/n_words.p', 'wb'))

    #remove the review with zero length from the reviews_ints list.
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    labels = np.array([labels[ii] for ii in non_zero_idx])

    #create feature vector
    features = np.zeros((len(reviews_ints), seq_len), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_len]

    #split the data into training, validation, and test sets
    split_idx = int(len(features) * split_frac)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    test_idx = int(len(val_x) * 0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    pickle.dump((train_x, train_y), open('preprocess_data/train.p', 'wb'))
    pickle.dump((val_x, val_y), open('preprocess_data/validation.p', 'wb'))
    pickle.dump((test_x, test_y), open('preprocess_data/test.p', 'wb'))

    # print(type(train_y))
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))
