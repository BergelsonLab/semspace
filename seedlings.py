import csv
import sys
import os
import pandas
from words import *


class Seedlings(object):
    def __init__(self, bl_path):
        self.data = self.load_data(bl_path)

    def load_data(self, path):
        df = pandas.read_csv(path)
        df.basic_level = df.basic_level.str.lower()
        df['count'] = df.groupby('basic_level')['basic_level'].transform('count')
        return df

    def wordmap(self):
        wordmap = {}
        for month in self.data['month'].unique():
            month_words = self.data.basic_level[self.data.month == month].dropna()
            month_words = month_words[month_words != "***fix me***"]
            wordmap[month] = month_words
        all_words = self.data.basic_level.dropna()
        all_words = all_words[all_words != "***fix me***"]
        wordmap['all'] = all_words
        return wordmap


if __name__ == "__main__":

    bl_file = sys.argv[1]

    # glove = GloVe(vocab="data/model/dict_glove_42b_300",
    #               vectors="data/model/vectors_glove_42b_300.npy")

    seedlings = Seedlings(bl_file)
    seedlings_wordmap = seedlings.wordmap()
    # seedlings_wordmap = load_concat_basic_level(bl_file)


    # glove.graph_cosine_range(output_path="seedlings2", wordmap=seedlings_wordmap,
    #                          start=0.4, end=0.41, step=0.01)
    print
