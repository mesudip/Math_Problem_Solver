#!/usr/bin/python3
import nltk
from nltk import PunktSentenceTokenizer
from networkx import MultiDiGraph
from networkx import draw_networkx
from matplotlib import pyplot as plot
from networkx import set_node_attributes
from collections import deque


class Classifier:
    classification_step = ('Sentence', 'Noun Phrase', 'Entity')

    @staticmethod
    def sentence_classifier(input_query):
        '''
        For given plain english text, performs word tagging and separates the sentences in the text
        :param input_query: str object containing text
        :return: nltk.tree.Tree structure that contains sentences and tagged words.
        '''
        sentences = PunktSentenceTokenizer().tokenize(input_query)

        sentence_chunk = nltk.tree.Tree("Query", [])
        # conjunction chunk.
        for sentence in sentences:
            tokenized = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokenized)
            sentence_chunk.append(nltk.tree.Tree('Sentence', tagged))
        return sentence_chunk

    @staticmethod
    def verb_classifier(chunk):
        '''
        Separates verb and other phrases from the sentence.
        :param chunk: the chunk to classify. It must  be of type sentence.
        :return: the classified one is returned.
        '''
        chunk_regex = r'''Punctuation:{<\.>+}'''
        parser = nltk.RegexpParser(chunk_regex)
        chunk = parser.parse(chunk)
        chunk_regex = r"""Noun Phrase:{<.*>+}
                                        }(<VB.?|CC|MD>+)|<Punctuation>+{"""
        parser = nltk.RegexpParser(chunk_regex)
        ans = parser.parse(chunk)
        chunk_regex = r"""Verb Phrase:{<VB.?|CC|MD>+}"""
        parser = nltk.RegexpParser(chunk_regex)
        return parser.parse(ans)

    @staticmethod
    def entity_classifier(chunk):
        '''
        Separates all the entities separated by preposition.
        :param chunk: the chunk to classify. It must be of type Noun Phrase for correct interperation.
        :return: the clssified one is returned.
        '''
        chunk_regex = r'''Entity: {<.+>+}
                            }<IN|TO>{'''
        parser = nltk.RegexpParser(chunk_regex)
        chunk = parser.parse(chunk)
        chunk_regex = r'''Preposition:{<IN|TO>}'''
        parser = nltk.RegexpParser(chunk_regex)
        return parser.parse(chunk)

    @staticmethod
    def noun_classifier(chunk):
        '''
        Separates the main noun from description + noun form.
        :param chunk: the chunk to classify. It must be of type Entity
        :return: the classified one is returned.
        '''
        chunk_regex = r'''Attribute:{<.+>+}
                                            }<NN.*>+{ '''
        parser = nltk.RegexpParser(chunk_regex)
        ans = parser.parse(chunk)
        chunk_regex = r'''Noun:{<NN.*|P>+} '''
        parser = nltk.RegexpParser(chunk_regex)
        return parser.parse(ans)

    classify_function = ()

    @staticmethod
    def direct_classifier(chunk, classify_type, classifier):
        '''
        Chunks the 'chunks of classify_type' by using classifier function.
        :param chunk: the chunk to classify. It must be a nltk.tree.Tree structure.
        :param classifier: the classifier to use
        :param classify_type: the label of the chunk that will be classified.
        :return: returns new weak reference copy of classified_chunk.
        '''
        stack = [chunk, ]
        while len(stack) is not 0:
            chunk = stack.pop()
            for i in range(len(chunk)):
                if type(chunk[i]) is nltk.tree.Tree:
                    if chunk[i].label() == classify_type:
                        chunk[i] = classifier(chunk[i])
                    else:
                        stack.append(chunk[i])

    @staticmethod
    def construct_tree(input_str):
        sentences = Classifier.sentence_classifier(input_str)
        for _function in Classifier.classify_function:
            Classifier.direct_classifier(sentences, *_function)
        return sentences

    @staticmethod
    def find_all(chunk, type_name_to_find):
        found = deque()
        stack = [chunk]
        while len(stack) is not 0:
            chunk = stack.pop()
            if type(chunk) is nltk.tree.Tree:
                if chunk.label() == type_name_to_find:
                    found.appendleft(
                        (tuple([a[0] for a in chunk.leaves()]) if len(chunk.leaves()) > 1 else chunk.leaves()[0][0],
                         {'context': [chunk]})
                    )
                else:
                    stack.extend(chunk)
        print(found)
        return found

    @staticmethod
    def format_beautify(tagged_sentences):
        regex='''proper_noun {<(NNP|NNS)>}'''

Classifier.classify_function = (('Sentence', Classifier.verb_classifier),
                                ('Noun Phrase', Classifier.entity_classifier),
                                ('Entity', Classifier.noun_classifier),
                                )


class WorldContext(MultiDiGraph):
    def draw(self):
        draw_networkx(self)
        plot.show()

    pass

    def set_attribute(self, key, *values):
        set_node_attributes(self, key, tuple(values))


if __name__ == "__main__":
    sentence = "At what rate percent per annum will $4000 yield an interest of $410 in 2 years?"
    tree = Classifier.construct_tree(sentence)
    tree.draw()
    context = WorldContext()

    for i in Classifier.find_all(tree, "Noun"):
        print("Processing", i[0])
        for _tree in i[1]['context']:
            print("reading", _tree)
            if type(_tree) is nltk.tree.Tree:
                if _tree.label() == 'Noun':
                    print("\tFound ", _tree)
                    context.add_node(tuple([a[0] for a in _tree.leaves()]) if len(_tree.leaves()) > 1 else _tree[0][0])
    context.draw()
# tree.draw()
