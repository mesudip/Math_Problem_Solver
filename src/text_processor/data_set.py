#!/usr/bin/python3
import json
import numpy as np
import sys
import sqlite3
import nltk
from collections import Counter
from text_processor import equation_processor
from array import array

# from text_processor.classifier import Classifier


def database_init_sqlite(source='../../dataset/dataset.json', destination='../../dataset/dataset.sqlite3'):
    database = sqlite3.connect(destination)
    cursor = database.cursor()
    cursor.execute('DROP TABLE IF EXISTS dataset')
    cursor.execute('''
                    CREATE TABLE 
                            dataset(
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                equation_count INTEGER,
                                solution_count INTEGER,
                                question TEXT,
                                equation TEXT,
                                solution INTEGER
                            )
                   ''')
    json_data = json.loads(open(source).read())
    proper_data = []
    for data in json_data:
        if 'lEquations' in data and 'lSolutions' in data and 'sQuestion' in data:
            data['iequation'] = len(data['lEquations'])
            data['isolution'] = len(data['lSolutions'])
            data['lEquations'] = str(data['lEquations'])
            data['lSolutions'] = str(data['lSolutions'])
            proper_data.append(data)
    cursor.executemany('''
        INSERT INTO dataset(question,equation,solution,solution_count,equation_count) VALUES(:sQuestion,:lEquations,:lSolutions,:isolution,:iequation)
    ''', proper_data)

    database.commit()
    cursor.close()
    database.close()


def database_preprocessed_init_sqlite(source='../../dataset/dataset_preprocessed.json',
                                      destination='../../dataset/dataset_preprocessed.sqlite3'):
    database = sqlite3.connect(destination)
    cursor = database.cursor()
    cursor.execute('DROP TABLE IF EXISTS dataset')
    cursor.execute('''
                        CREATE TABLE 
                                dataset(
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    param_count INTEGER,
                                    digits TEXT,
                                    question TEXT,
                                    equation TEXT,
                                    solution TEXT,
                                    verbs TEXT
                                )
                       ''')
    json_data = json.loads(open(source).read())
    for data in json_data:
        data['iparam'] = len(data['digits'])
        data['digits'] = str(data['digits'])
        data['verbs'] = str(data['verbs'])
    cursor.executemany('''
            INSERT INTO dataset(question,equation,solution,param_count,digits,verbs) 
            VALUES(:question,:equation,:solution,:iparam,:digits,:verbs)
        ''', json_data)

    database.commit()
    cursor.close()
    database.close()


def read_data_sqlite(source='../../dataset/dataset.sqlite3'):
    database = sqlite3.connect(source)
    ans = database.execute('SELECT * FROM dataset')
    return list(ans)


def read_data_json(source='../../dataset/dataset.json'):
    json_data = json.loads(open(source).read())
    proper_data = []
    for data in json_data:
        if 'lEquations' in data and 'lSolutions' in data and 'sQuestion' in data:
            proper_data.append(
                {'question': data['sQuestion'], 'equation': data['lEquations'], 'solution': data['lSolutions']})
    return proper_data


def read_preprocessed_data_json(source='../../dataset/dataset_preprocessed.json'):
    return json.loads(open(source).read())


def _get_preprocessed_data():
    # the lemmatizing function to be used for converting all forms of verbs into same form.
    # eg. is, am , was --> be
    #     eats, ate   --> eat
    lemmatize = nltk.stem.WordNetLemmatizer().lemmatize

    # read the json data : it's a list of dictionaries containing data.
    # each dictionary has 'question', 'equation' and 'solution'
    # we now take only those whose  equaiton and solution count is 1.
    json_data = [x for x in read_data_json() if (len(x['equation']) is 1) and (
        len(x['solution']) is 1)]  # TODO 'Make it to work with multiple equations and solutions'

    # for each questions in data set.
    error_count = 0
    best_data = []
    all_verbs = set()
    all_equations=set()
    for data in json_data:
        # the solution is previously a list having only one element
        # so change that
        data['solution'] = data['solution'][0]

        # tokenize the question by sentences
        sentences = nltk.sent_tokenize(data['question'])

        # to store word tokens

        tokens = []
        for sentence in sentences:
            # tokenize words in sentence
            words = nltk.word_tokenize(sentence)
            # add the part of speech info to the tokenized word and append all of the words into the tokens list
            tokens.extend(nltk.pos_tag(words))

        # find verbs in the word list and make it's counter.
        data['verbs'] = Counter([lemmatize(verb[0], pos='v') for verb in tokens if verb[1].startswith('V')])
        all_verbs.update(data['verbs'].keys())
        try:
            # now extract the digits in the question
            digits = data['digits'] = tuple([float(token[0]) for token in tokens if token[1] == 'CD'])
            _set = set(digits)
            if len(_set) is not len(digits):
                raise Exception("Duplicate number in question", "Number matching is not possible")
            # then convert equation into general form.
            data['equation'] = equation_processor.generalize(data['equation'][0], data[
                'digits'])  # TODO 'Change it when we model multiple equation and solution'
            best_data.append(data)
            all_equations.add(data['equation'])
        except Exception as e:
            # error formating the data
            # just print error informaiton and ignore the error
            # print(e, file=sys.stderr)
            # print(data, file=sys.stderr)
            # print(tokens, file=sys.stderr)
            # print()
            error_count += 1
            # remove the erronous data from the json list
    print('\n\n')
    print("Error data :", error_count)
    print("Correct data:", len(best_data))

    all_verbs=list(all_verbs)
    all_verbs.sort()
    all_equations=list(all_equations)
    all_equations.sort()

    return best_data,all_verbs,all_equations
class net_feeder:
    def __init__(self,all_verbs,all_equations):

        # assign some key for each verb.
        self.verb_index={}
        for index,verb in enumerate(all_verbs):
            self.verb_index[verb]=index

        self.equation_index={}
        for index, equation in enumerate(all_equations):
            self.equation_index[equation]=index

    def format_data_set(self,data_set):
        feeds=[]
        outputs=[]

        for data in data_set:

            feed=array((0.,)*len(self.verb_index))
            output=array((0.,)*(len(self.equation_index)+1))

            verbs_of_data=data['verbs']
            for verb in verbs_of_data:
                feed[self.verb_index[verb]] = verbs_of_data[verb]

            output[self.equation_index[data['equation']]]=1.0
            bits = [float(x) for x in bin(len(data['digits']))[2:]]
            start=-1

            for bit in bits:
                output[start]=i
                start-=1

            feeds.append(feed)
        return feeds


if __name__ == "__main__":
    data,verbs,equations = _get_preprocessed_data()

    json.dump(data, open('../../dataset/dataset_preprocessed.json', 'w'))
    database_preprocessed_init_sqlite()
    print(verbs)
    print("Verb count:",len(verbs))
# jsondata = 0
# training_set = jsondata[:500]
#
# test_set = jsondata[3500:3600]
#
# questions = []
# answers = []
# for i in range(len(training_set)):
#     questions.append(training_set[i]['sQuestion'])
#     answers.append(int(float(training_set[i]['lSolutions'][0])) % 100)
#
# intquestions = np.zeros([len(questions), 800])
# for i in range(len(questions)):
#     for j in range(len(questions[i])):
#         intquestions[i][j] = ord(questions[i][j])
#
# intanswers = np.zeros([len(answers), 100])
# for i in range(len(answers)):
#     intanswers[i][answers[i]] = 1
#
# train_x = np.array(intquestions)
# train_y = np.array(intanswers)
#
# for i in range(len(test_set)):
#     questions.append(test_set[i]['sQuestion'])
#     answers.append(int(float(test_set[i]['lSolutions'][0])) % 100)
#
# intquestions = np.zeros([len(questions), 800])
# for i in range(len(questions)):
#     for j in range(len(questions[i])):
#         intquestions[i][j] = ord(questions[i][j])
#
# intanswers = np.zeros([len(answers), 100])
# for i in range(len(answers)):
#     intanswers[i][answers[i]] = 1
#
# test_x = np.array(intquestions)
# test_y = np.array(intanswers)
# print(test_x)
# print(test_y)
