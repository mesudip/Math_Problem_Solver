#!/usr/bin/python3

from text_processor.data_set import Net_feeder
import traceback
import tensorflow as tf
import random
import numpy as np
from tensorflow.contrib.learn import DNNClassifier as net
import sys

feeder = Net_feeder.load_feed_from_file()
_input, _output = feeder.get_vectors_from_extended_data_set()

in_count = len(_input[0])
out_count = max(_output) + 1

print()
print()
print()

print("the no of inputs:", in_count)
print("the no of outputs:", out_count)
# net = NeuralNet(in_count, in_count, in_count, out_count)
#
# cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=net.train_output, logits=net.output))
# optimizer=tf.train.AdamOptimizer().minimize(cost_function)
# net.train(_input, _output, cost=cost_function,epoches=1000,optimizer=optimizer)
# ans=net.stimulate(*_input[0])

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=in_count)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
ai_net = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[in_count, in_count - 50, in_count - 50],
                                        n_classes=out_count)
# model_dir=""

# shuffle the fucking thing first
combined = list(zip(_input, _output))
random.shuffle(combined)
_input[:], _output[:] = zip(*combined)

bak_input = _input
bak_output = _output
_input = np.array(_input[:-50], dtype=np.float32)

_output = np.array(_output[:-50], dtype=np.int32)

print("Begin Training")
ai_net.fit(_input, _output, steps=10000)

print("End training")
# Evaluate accuracy.
test_input = np.array(bak_input[-50:], dtype=np.float32)

test_output = np.array(bak_output[-50:], dtype=np.int32)

accuracy_score = ai_net.evaluate(test_input,
                                 test_output
                                 )['accuracy']

print('Accuracy: {0:f}'.format(accuracy_score))

while True:
    try:
        print("Give a question :")
        user_input = input()

        if user_input == 'exit':
            break;

        net_input, params, verbs = feeder.format_question(user_input)
        if len(params) < 2:
            raise ValueError("That question is not arithmetic")
        print("Feature Length :", len(net_input))
        print("Detected parameters :", params)
        print("Detected Verbs :", verbs)
        ret = ai_net.predict(x=np.array([net_input], dtype=np.float32),as_iterable=False)[0]
        print("Obtained class :",ret)
        print("Predicted Equation :", feeder.get_equation_from_index(ret))
        print()
        print()
    except Exception as e:
        print("\n\nError while parsing question\n",e.args,'\n\n',file=sys.stderr)

# print(ans)
# print([round(x) for x in list(ans[0])])
