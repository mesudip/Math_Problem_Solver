from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import data_set as data_set
import question_preprocessor
import sys
# # Data sets
# IRIS_TRAINING = "iris_training.csv"
# IRIS_TEST = "iris_test.csv"

# Load datasets.
# training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TRAINING,
#     target_dtype=np.int,
#     features_dtype=np.float32)
# test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#     filename=IRIS_TEST,
#     target_dtype=np.int,
#     features_dtype=np.float32)

training_set_x=data_set.train_x

training_set_y=data_set.train_y

test_set_x=data_set.test_x
test_set_y=data_set.test_y


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=329)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[330, 330, 330],
                                            n_classes=13)
                                            #model_dir=""

#Fit model.
classifier.fit(x=training_set_x,
               y=training_set_y,
               steps=1000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set_x,
                                     y=test_set_y)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
print(accuracy_score)



def test(val):

        y = list(classifier.predict(np.array([data_set.normalized_questions[val]],dtype=int), as_iterable=True))
        print(data_set.questions[val])
        print(data_set.training_set[val]["equation_general"])

        print('Predictions:'+str(y))

        found=None
        for a in data_set.training_set:
            if (a['equation_class'] == y[0]):
                found=a
                break

        # listt=data_set.training_set[:]['equation_general']
        # value=listt.index(y)
        # print(data_set.training_set[value]['equation_general'])
        if found is None:
            print("You get nothing.")
        print(found['equation_general'])



for i in range(1700,1747):
    test(i)

#
# inp="Ram has 5 apples. He ate 3 of them. How many apples does he have now?"
# print(question_preprocessor.process([inp]))
#
# sys.exit(0)
# #
# # while True:
# #     question=question_preprocessor.process([inp])
# #     test_direct(question)
# #     inp=input()
