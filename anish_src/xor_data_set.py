#!/usr/bin/python3
from itertools import product
import numpy as np
import itertools
pos=[0,1]

#for a in product((0,1),(0,1),(0,1)):
train_x=[]
train_y=[]
for a in product(pos,repeat=4):
    for b in product(pos,repeat=4):
        train_x.append([x for x in itertools.chain(a, b)])
        train_y.append([ x[0]|x[1] for x in   zip(a,b)])




test_x=train_x
test_y=train_y


if __name__=="__main__":

    print(training_set)