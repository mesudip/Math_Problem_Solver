import json
import numpy as np
import random
import question_preprocessor


#dataset import here

with open("dataset_with_equation_class.json",'r') as dataset_file:

    data=dataset_file.read()

    jsondata=json.loads(data)


#sorting according to the equation class
sorted_dataset=sorted(jsondata,key=lambda x: x['equation_class'])


#selecting top 13 equation classes. No. of data in the top 13 equation set = 1747 (this number is subject to change)
training_set=sorted_dataset[:1747]


#randomizing the data
random.shuffle(training_set)

#extracting questions and answers from the data
questions=[]
answers=[]

for i in range (len(training_set)):
    questions.append(training_set[i]['sQuestion'])
    answers.append(training_set[i]['equation_class'])


#normalized questions (i.e. question represented according to the verbs)

normalized_questions=question_preprocessor.process(questions)
train_x=np.array(normalized_questions[:1600],dtype=int)


#
# intanswers=np.zeros([len(answers),13])
# for i in range(len(answers)):
#     intanswers[i][answers[i]]=1

train_y=np.array(answers[:1600],dtype=int)


test_x=np.array(normalized_questions[1600:1700],dtype=int)
test_y=np.array(answers[1600:1700],dtype=int)



