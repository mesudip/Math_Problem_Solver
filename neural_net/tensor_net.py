#!/usr/bin/python3
from tensorflow import float32
from tensorflow import placeholder
from itertools import islice
from tensorflow import Variable as variable
from tensorflow import random_normal
from tensorflow import matmul, add, sigmoid
from tensorflow import Session as session
from tensorflow import global_variables_initializer
import numpy as np
import tensorflow as tf
from tensorflow import Print as tf_print
import array


class NeuralNet:
    def __init__(self, *net_structure, input_type=float32, output_type=float32, weight_mean=0, weight_stddev=0.03,
                 activation=sigmoid, _session=None):
        '''
        Creates a neural net structure in tensorflow
        :param net_structure:
        :param input_type:
        :param output_type:
        :param weight_mean:
        :param weight_stddev:
        :param activation:
        '''
        self.initializer = global_variables_initializer()
        self.session = _session

        # the columns will contain the structure of the net.
        self.columns = tuple(net_structure)
        # the tensor to which the input values will be fed.
        self.receptor = placeholder(input_type, [None, self.columns[0]], name='neural_net_receptor')

        # the tensor to which we insert the expected output for input while training with dataset.
        self.train_output = placeholder(output_type, [None, self.columns[-1]], name='neural_net_expected_output')

        # initialize hidden layers as empty list.
        self.hidden_layer = []

        # the no of neurons in previous layer.
        # so for the i'th layer in the hidden layer, the input_size gives the input to each neuron in the layer.
        input_size = net_structure[0]

        # now for each hidden layer structure, construct the tensor variables to store the weights and bias.
        # the value of i in the iteration is the no of neurons in that layer.
        layer_no = 1
        for i in islice(net_structure, 1, len(net_structure)):
            # construct matrix for the current layer
            # the columns of the layer is for a neuron.
            # each row of a column has i'th weight value for the neuron.
            weights = variable(random_normal([input_size, i], stddev=weight_stddev, mean=weight_mean),
                               name="hidden_layer_weight_"+str(layer_no))

            # the bias value for each neron
            bias = variable(random_normal([1, i]), name='hidden_layer_bias_'+str(layer_no))

            # now for the next layer, the no of input to the neurons is the no of neurons in this layer.
            input_size = i

            # append the weight and bias to the hidden_layer list
            self.hidden_layer.append((weights, bias))

            #increase the layer_no counter
            layer_no+=1

        # declare how the neurons in each layer are connected.

        previous_output = self.receptor
        for layer in self.hidden_layer:
            # the result by multiplying consecutive weights and input for all neurons in a layer
            _add = add(matmul(previous_output, layer[0]), layer[1])

            # using activation function for the obtained output from each neuron in layer
            activated = activation(_add)

            # previous_output to be save for making it as input in another layer.
            previous_output = activated

        # the output layer is the returned value of the last hidden layer.
        self.output = previous_output

        # initialize all the variables
        if self.session is None:
            self.session = session()
        self.session.run(global_variables_initializer())

    def stimulate(self, *args):

        try:
            # if the provided input is not a single input but a series of inputs, we need something else for this
            # if the provided input is only a single input then the iter() function will cause an excpetion.
            iter(args[0])
            input = np.array(args)
            # check that the no. of inputs is correct for each input
            if len(input[0]) is not self.columns[0]:
                raise Exception("Invalid no of inputs to the neural net.",
                                "Expected %d arguments got %d." % (self.columns[0], len(args)))

        # if the provided input is a single set of inputs to receptor
        except TypeError as e:
            # check if the provided no of arguments match with the no of inputs to the neural net.
            if len(args) is not self.columns[0]:
                raise Exception("Invalid no of inputs to the neural net",
                                "Expected %d arguments got %d" % (self.columns[0], len(args)))
            input = np.array([args])

        # run the tensor graph for neuralNet and return the result
        print (input)
        return self.session.run(self.output, {self.receptor: input})

    def __str__(self):
        if self.session is None:
            return "Neural Net cannot be converted to str without a active session"
        # the string to which the str values will be appended
        __out = ''

        # variable to keep track of the hidden layer no.
        i = 1
        # add info about the hidden layer structure of the neural net.
        __out += 'Neural Net ' + str(self.columns) + ':\n'

        # iterate for each layer in the hidden layer.
        for weights, bias in self.hidden_layer:
            # the i'th hidden lyaer.
            __out += '  Hidden Layer %d :\n' % (i)

            # the structures for keeping the weight and bias are row matrices. transpose them to get column matrix
            # the session.run method is used to return the values of weights in the tensor.
            bias = np.transpose(self.session.run(bias))
            weights = np.transpose(self.session.run(weights))

            # variable to keep track of the no of neuron in each layer.
            j = 1

            # iterate for each neuron in the hidden layer
            for neuron_weights, neuron_bias in zip(weights, bias):
                __out += "    Neuron {} < bias : {}\tweights : {} >\n".format(j, neuron_bias, neuron_weights)
                j += 1

            # now turn of next hidden layer. so increment i.
            i += 1
        # return the str with the neural net description.
        return __out

    def train(self, input_array, output_array):
        # convert them into numpy array. Obvious line.
        input_array=np.array(input_array)
        output_array=np.array(output_array)

        # cost function to calculate the difference between the obtained and expected output.
        # this function is for classification problem only. may be use standard deviation function? don't know w
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_output,logits=self.output ))

        # found better way to get the standard error function.
        # cost =sum(
        #           {expected(i) - obtained(i)}2
        #                       : for all i'th output of neural net.
        #           )
        cost=tf.reduce_sum(tf.square(tf.subtract(self.train_output,self.output)))


        # the optimizer to use. Don't have any idea what it does internally.
        optimizer = tf.train.AdamOptimizer().minimize(cost)


        # initilize all the variables again.
        # This will reinitialize the weights and bias of the neurons in neural net too.
        self.session.run(global_variables_initializer())

        # the no of times we will train the net.
        epoches = 10
        # now train the net.
        for _ in range(epoches):
            optimized, cost = self.session.run([optimizer, cost],
                                               feed_dict={self.receptor: input_array,self.train_output:output_array})



input_data = [
         [1.0, 1.0],
         [1.0, 0.0],
         [0.0, 1.0],
         [0.0, 0.0],
         ]
output_data= [
    [1.0],
    [1.0],
    [1.0],
    [0.0],
]

if __name__ == "__main__":
    net = NeuralNet(2,2,1)
    print(net)

    print(net.stimulate(*input_data))
    print(net.stimulate(1,1))

    #net.train(input_data,output_data)

    net.session.close()
