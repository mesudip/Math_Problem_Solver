#!/usr/bin/python3
import json
import numpy as np
import sys
import sqlite3
import nltk
from text_processor.classifier import  Classifier

def database_init(source='../../references/dataset.json', destination='../../references/dataset.sqlite3'):
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


def read_data_sqlite(source='../../references/dataset.sqlite3'):
    database = sqlite3.connect(source)
    ans = database.execute('SELECT * FROM dataset')
    return list(ans)


def read_data_json(source='../../references/dataset.json'):
    json_data = json.loads(open(source).read())
    proper_data = []
    for data in json_data:
        if 'lEquations' in data and 'lSolutions' in data and 'sQuestion' in data:
            proper_data.append([data['sQuestion'], data['lEquations'], data['lSolutions']])
    return proper_data


def get_preprocessed_data():
    json_data=read_data_json()
    for data in json_data:
        tagged=Classifier.sentence_classifier(data[0])
        for sentence, context in Classifier.find_all(tagged,"Sentence"):
            print(sentence)

        input()



if __name__=="__main__":
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
