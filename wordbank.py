import sys
import os

import pandas


filter_categories = [
    "action_words", "games_routines",
    "question_words", "pronouns",
    "descriptive_words", "helping_verbs",
    "connecting_words", "quantifiers",
    "time_words", "locations"
]

filter_types = [
    "complexity", "word_endings_verbs",
    "word_forms_verbs", "combine",
    "how_use_words"
]

class WordBank(object):

    def __init__(self, input):
        self.input_file = input
        self.load_file(input)

    def load_file(self, input):
        self.data = pandas.read_csv(input)

    def top_n_month(self, month, n):
        result = self.data.sort_values(by=month, ascending=False).tolist()
        return result[:n]

    def sorted(self, month):
        return self.data.sort_values(by=month, ascending=False)

    def wordmap(self):
        wordmap = {}
        dataset_name = os.path.basename(self.input_file).replace(".csv", "")

        words = []
        for index, word in self.data.iterrows():
            if word['category'] not in filter_categories:
                if word['type'] not in filter_types:
                    the_word = filter_parens(word['definition'])
                    if "/" in the_word:
                        split_word = the_word.split('/')
                        for element in split_word:
                            element = element.replace("*", "")
                            words.append(element)
                    else:
                        the_word = the_word.replace("*", "")
                        words.append(the_word)

        wordmap[dataset_name] = pandas.Series(words)
        return wordmap


def filter_parens(word):
    index = word.find("(")
    if index != -1:
        return word[:index]
    else:
        return word

if __name__ == "__main__":

    input_file = sys.argv[1]

    wordbank = WordBank(input_file)

    wordmap = wordbank.wordmap()

    print