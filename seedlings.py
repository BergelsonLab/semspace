import csv
import sys
import os
import pandas
from words import *


class Seedlings(object):
    def __init__(self, bl_path):
        self.data = pandas.read_csv(bl_path)

    def wordmap(self):
        wordmap = {}
        for month in self.data['month'].unique():
            month_words = self.data.basic_level[self.data.month == month].dropna()
            month_words = month_words[month_words != "***FIX ME***"]
            wordmap[month] = month_words
        return wordmap



# def load_basic_levels(path, wordmap):
#     """
#     wordmap is a dict of month to list-of-words mappings
#
#     :param path:
#     :param wordmap:
#     :return:
#     """
#     key = int(os.path.basename(path)[:2])
#     words = pandas.read_csv(path)["word"]
#     if key not in wordmap:
#         wordmap[key] = set(words.tolist())


# def load_seedlings(path):
#     wordmap = {}
#
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             load_basic_levels(os.path.join(root, file), wordmap)

if __name__ == "__main__":

    bl_file = sys.argv[1]

    glove = GloVe(vocab="data/model/dict_glove_42b_300",
                  vectors="data/model/vectors_glove_42b_300.npy")

    seedlings = Seedlings(bl_file)
    seedlings_wordmap = seedlings.wordmap()
    # seedlings_wordmap = load_concat_basic_level(bl_file)


    glove.graph_cosine_range(output_path="seedlings2", wordmap=seedlings_wordmap,
                             start=0.4, end=0.41, step=0.01)
    print
