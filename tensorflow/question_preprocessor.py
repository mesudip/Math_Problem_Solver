import numpy as np

import nltk
from nltk import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

#different forms of verbs
useful_pos=['VB','VBD','VBG','VBN','VBP','VBZ']

lemma=WordNetLemmatizer().lemmatize

def process(questions):

    #sentence tokenizing (sent_tokenizer better?)

    sentences_list = []
    for question in questions:
        sentences_list.append(PunktSentenceTokenizer().tokenize(question))


    # tokenization and tagging with lemmatization
    words_list = []

    for sentences in sentences_list:
        words = []
        for sentence in sentences:
            words.append([lemma(word, pos='v') for word in nltk.word_tokenize(sentence) if
                      word not in stopwords.words('english')])

        tagged_words = [nltk.pos_tag(word) for word in words]
        words_list.append(tagged_words)



    # verb categorization (separating verbs from the words)
    important_words = []

    for words in words_list:
        for word in words:
            for x in word:
                if ((x[1] in useful_pos) and (x[0] not in important_words)):
                    important_words.append(x[0])


    # normalizing question set
    question_array = np.zeros([len(questions), len(important_words)])

    for i in range(len(words_list)):
        for words in words_list[i]:
            for word in words:
                if (lemma(word[0],pos='v')) in important_words:
                    question_array[i][important_words.index(word[0])] += 1


    return question_array

