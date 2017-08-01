import tensorflow as tf


def main(x, y):
    x1 = tf.constant(x)
    x2 = tf.constant(y)
    result = tf.multiply(x1, x2)
    with tf.Session() as sess:
        x3 = sess.run(result)
    return x3
