import tensorflow as tf
import data_set as data_set
import numpy as np


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
#n_nodes_hl4 = 100


n_classes =13
batch_size = 1
question_length=329
# input feature size = 28x28 pixels = 784
x = tf.placeholder(tf.int32, [None, question_length])
y = tf.placeholder(tf.int32)


def neural_network_model(data):
    # input_data * weights + biases
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([question_length, n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    # hidden_l4 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
    #              'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(tf.cast(data,tf.float32), hidden_l1['weights']), hidden_l1['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases'])
    l2 = tf.nn.sigmoid(l2)

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases'])
    l3 = tf.nn.sigmoid(l3)

    # l4 = tf.add(tf.matmul(l3, hidden_l4['weights']), hidden_l4['biases'])
    # l4 = tf.nn.sigmoid(l4)

    output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases'])
    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))  # v1.0 changes

    # optimizer value = 0.001, Adam similar to SGD
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)
    epochs_no = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # v1.0 changes

        # training
        for epoch in range(epochs_no):
            epoch_loss = 0
            for i in range(len(data_set.train_x)):
                epoch_x, epoch_y = data_set.train_x[i:i+batch_size],data_set.train_y[i:i+batch_size]
                i, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                # code that optimizes the weights & biases
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)

        # testing
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: data_set.test_x, y: data_set.test_y}))


train_neural_network(x)