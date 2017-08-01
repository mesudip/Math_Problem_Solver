#!/usr/bin/python3

import random
import sys
import numpy as np

import tensorflow as tf

from text_processor.data_set import Net_feeder
from text_processor.data_set import Preprocessor



# the **cking warning messages can be discarded into the null file.



print("Initializing Neural Net ...")
preporcessor = Preprocessor.get_from_file('dataset.sqlite3')

data, verbs, equations = preporcessor.get_best_data_set(16)
feeder = Net_feeder(verbs, equations)

_input, _output = feeder.get_vectors_from_data_set(data)

in_count = len(_input[0])
out_count = max(_output) + 1

print("the no of inputs in neural net :", in_count)
print("the no of classes in neural net:", out_count)
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
                                        hidden_units=[in_count, in_count + 200, in_count + 200],
                                        n_classes=out_count)
# model_dir=""

# shuffle the fucking thing first
combined = list(zip(_input, _output))
random.shuffle(combined)
_input[:], _output[:] = zip(*combined)

print("Length of training data:", len(_input))

_input = np.array(_input, dtype=np.float32)

_output = np.array(_output, dtype=np.int32)

print("Begin Training :")
for i in range(50, 900,50):
    ai_net.fit(_input, _output, steps=50)
    print("No of complete iterations in training :", i)
print("End training")
accuracy_score = ai_net.evaluate(_input,
                                 _output
                                 )['accuracy']
print('Training Accuracy: {0:f}'.format(accuracy_score))


def get_result(user_input):
    try:
        result = {}
        net_input, params, verbs = feeder.format_question(user_input)
        if len(params) < 2:
            raise ValueError("That question doesn't have enough parameters for mathematics.")
        result['digits'] = params
        result['verbs'] = dict(verbs)

        ret = ai_net.predict(x=np.array([net_input], dtype=np.float32), as_iterable=False)[0]
        equation = feeder.get_equation_from_index(ret)
        result["equation :"] = equation
        try:
            N0 = params[0];
            N1 = params[1];
            N2 = params[2];
            N3 = params[3];
            N4 = params[4];
        except Exception as e:
            pass
        ans = eval(equation.split('=')[-1])
        result['answer'] = ans
    except Exception as e:
        result['error'] = "Error while parsing question :" + str(e.args)
    return result

if __name__=="__main__":
    get_result('something')
