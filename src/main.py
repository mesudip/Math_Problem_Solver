#!/usr/bin/python3

from text_processor.data_set import Net_feeder
from neural_net.tensor_net import NeuralNet
import tensorflow as tf

feeder = Net_feeder.load_feed_from_file()
_input, _output = feeder.get_vectors_from_extended_data_set()

in_count = len(_input[0])
out_count = len(_output[0])

net = NeuralNet(in_count, in_count, in_count, out_count)

cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=net.train_output, logits=net.output))
optimizer=tf.train.AdamOptimizer(learning).minimize(cost_function)
net.train(_input, _output, cost=cost_function,epoches=1000,optimizer=optimizer)
ans=net.stimulate(*_input[0])
print(ans)
print([round(x) for x in list(ans[0])])


