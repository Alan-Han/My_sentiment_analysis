#!/usr/bin/python3

import numpy as np
import pickle
import tensorflow as tf

from My_sentiment_analysis import input_dataset

def train():
    # define the hyperparameters
    lstm_size = 256
    lstm_layers = 1
    batch_size = 500
    learning_rate = 0.001

    n_words = pickle.load(open('preprocess_data/n_words.p', 'rb'))
    n_words = n_words + 1  # Adding 1 because we use 0's for padding, dictionary started at 1

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Size of the embedding vectors (number of units in the embedding layer)
    embed_size = 300

    with graph.as_default():
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    with graph.as_default():
        # basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)

        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)

        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

    with graph.as_default():
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                 initial_state=initial_state)

    # output
    with graph.as_default():
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(labels_, predictions)

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # validation accuracy
    with graph.as_default():
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Batches
    def get_batches(x, y, batch_size=100):
        n_batches = len(x) // batch_size
        x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii + batch_size], y[ii:ii + batch_size]

    train_x, train_y = pickle.load(open('preprocess_data/train.p', 'rb'))
    val_x, val_y = pickle.load(open('preprocess_data/validation.p', 'rb'))

    # Training
    epochs = 10

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)

            for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(loss))

                if iteration % 25 == 0:
                    val_acc = []
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y in get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: x,
                                labels_: y[:, None],
                                keep_prob: 1,
                                initial_state: val_state}
                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.3f}".format(np.mean(val_acc)))
                iteration += 1
        saver.save(sess, "checkpoints/sentiment.ckpt")


def test():
    test_acc = []
    test_x, test_y = pickle.load(open('preprocess_data/test.p', 'rb'))
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))


def main():

    input_dataset.preprocess()

    train()

    test_model()


if __name__ == '__main__':
    main()