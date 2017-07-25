#!/usr/bin/python3
import json
import numpy as np
import sys
import sqlite3
import nltk
from collections import Counter

#from text_processor.classifier import Classifier


def database_init(source='../../dataset/dataset.json', destination='../../dataset/dataset.sqlite3'):
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


def get_preprocessed_data():
    # the lemmatizing function to be used for converting all forms of verbs into same form.
    # eg. is, am , was --> be
    #     eats, ate   --> eat
    lemmatize = nltk.stem.WordNetLemmatizer().lemmatize

    # read the json data : it's a list of dictionaries containing data.
    # each dictionary has 'question', 'equation' and 'solution'
    json_data = [x for x in read_data_json() if (len(x['equation']) is 1) and (len(x['solution']) is 1)]
    print(json_data)
    input()


    # for each questions in data set.
    for data in json_data:
        sentences = nltk.sent_tokenize(data['question'])
        tokens = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            tokens.extend(nltk.pos_tag(words))

        # find verbs in the word list and make it's counter.
        data['verbs'] = Counter([lemmatize(verb[0], pos='v') for verb in tokens if verb[1].startswith('V')])

        # now extract the digits in the question
        digits = data['digits'] = tuple([token[0] for token in tokens if token[1] == 'CD'])

        for i in range(len(digits)):
            for equation in data['equation']:
                equation.replace(str(digits[i]),'N_%d'%i)
                print(equation)
            input()

        print(data)
    return json_data


if __name__ == "__main__":
    get_preprocessed_data()
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
