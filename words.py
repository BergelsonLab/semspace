import array
import sys
import json
import operator
import os

import numpy as np
import pandas as pd

import seedlings
from wordbank import *


filter_words = ["and", "are", "it", "that",
                "a", "the", "how", "for",
                "on", "with", "you", "he",
                "she", "us", "we", "what",
                "why", "how", "all", "to",
                "very", "does", "from", "i",
                "do", "if", "in", "where",
                "is", "of", "at", "an", "only",
                "up", "who", "new", "just",
                "go", "high", "her", "give",
                "my", "me", "much", "goes",
                "saw", "many", "your", "let",
                "doing", "be", "than", "when",
                "way", "going", "get", "too",
                "one", "first", "over", "not",
                "like", "good", "out", "know",
                "but", "want", "so", "little",
                "top", "each", "little", "great",
                "big", "see", "away", "other",
                "as", "well", "whole", "perfect",
                "thing", "place", "looking", "make",
                "down", "take", "will", "really",
                "around", "nice", "say", "time",
                "best", "set", "large", "half",
                "look", "think", "front", "piece",
                "could", "come", "long", "favorite",
                "things", "pretty", "drop", "love",
                "stuff", "ready", "small", "sweet",
                "pull", "hot", "lot", "side", "no",
                "cool", "day", "end", "this", "through",
                "night", "or", "takes", "run", "coming",
                "about"
                ]
colors = [
    "black", "white", "red",
    "blue", "green", "brown",
    "yellow", "pink"
    ]
numbers = [
    "one", "two", "three",
    "four", "five", "six",
    "seven", "eight", "nine",
    "ten"
    ]




class SemanticGraph(object):
    def __init__(self, source, sim_func, thresh, path, wb):
        self.source = source
        self.sim_func = sim_func
        self.threshold = thresh
        self.path = path
        self.wb = wb
        self.load_graph()

    def load_graph(self):
        with open(self.path, "rU") as input:
            self.graph = json.load(input)

    def top_n_dense(self, n=0, all=False):
        if all:
            n = len(self.graph)
        top_words = top_n_words(self.graph, n)
        top_words = [(x[0], len(x[1])) for x in top_words]
        graph_df = pd.DataFrame(data=top_words, columns=["word", "edges"])
        result = pd.merge(graph_df, self.wb.data, left_on='word', right_on='definition')
        return result





def filter_plurals(words):
    results = []
    for word in words:
        if len(word) > 1:
            if word[-1] == "s":
                if word[:-1] in words:
                    continue
                else:
                    results.append(word)
            else:
                results.append(word)
    return results



class GloVe(object):
    def __init__(self, vocab, vectors):
        self.dim = None
        self.dict = None
        self.vectors = None
        self.load_model(vocab, vectors)

    def save_model(self, dict, vectors):
        np.save(vectors, self.vectors)
        with open(dict, "wb") as output:
            json.dump(self.dict, output)

    def load_model(self, dict, vectors):
        self.vectors = np.load(vectors)
        with open(dict, "rU") as input:
            self.dict = json.load(input)

    def graph_cosine_range(self, output_path,
                           wordmap, start, end,
                           step):
        for x in np.arange(start, end, step):
            generate_cosine_graphs(self, output_path, wordmap, x)

    def neighbor_density_cos(self, dx, word, corpus):
        if word not in self.dict:
            return None
        else:
            w1_vector = self.vectors[self.dict[word], :]
            distances = []

        for corp_word in corpus:
            if corp_word == word:
                continue
            for subword in corp_word.split("+"):
                if subword in self.dict:
                    if subword == word:
                        continue
                    w2_vector = self.vectors[self.dict[subword], :]

                    w1_vec_norm = np.zeros(w1_vector.shape)
                    d1 = (np.sum(w1_vector ** 2, ) ** (0.5))
                    w1_vec_norm = (w1_vector.T / d1).T

                    w2_vec_norm = np.zeros(w2_vector.shape)
                    d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
                    w2_vec_norm = (w2_vector.T / d2).T

                    dist = np.dot(w1_vec_norm.T, w2_vec_norm.T)

                    if 1-dist <= dx:
                        distances.append((subword, dist))
        return distances

    def neighbor_density_euclid(self, dx, word, corpus):
        if word not in self.dict:
            return None
        else:
            w1_vector = self.vectors[self.dict[word], :]
            distances = []

        for corp_word in corpus:
            if corp_word == word:
                continue
            for subword in corp_word.split("+"):
                if subword in self.dict:
                    if subword == word:
                        continue
                    w2_vector = self.vectors[self.dict[subword], :]

                    w1_vec_norm = np.zeros(w1_vector.shape)
                    d1 = (np.sum(w1_vector ** 2, ) ** (0.5))
                    w1_vec_norm = (w1_vector.T / d1).T

                    w2_vec_norm = np.zeros(w2_vector.shape)
                    d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
                    w2_vec_norm = (w2_vector.T / d2).T

                    dist = np.linalg.norm(w1_vec_norm.T - w2_vec_norm.T)

                    if dist <= dx:
                        distances.append((subword, dist))
        return distances


def load_glove_pretrain(path):
    dct = {}
    vectors = array.array('d')

    with open(path, 'rU') as input:
        for i, line in enumerate(input):
            tokens = line.split()

            word = tokens[0]
            entries = tokens[1:]

            dct[word] = i
            vectors.extend(float(x) for x in entries)

    dim = len(entries)
    no_vectors = len(dct)

    nparray = np.array(vectors).reshape(no_vectors,dim)
    glove = GloVe()
    glove.dict = dct
    glove.vectors = nparray
    glove.dim = dim
    return glove



def top_n_words(wordmap, n):
    wordlist = [(word, neighbors)
                for word, neighbors in wordmap.items()
                    if neighbors is not None
                ]

    wordlist.sort(key = lambda x: len(x[1]), reverse=True)
    return wordlist[:n]


def generate_cosine_graphs(model, path, wordmap, threshold):
    for key, value in wordmap.items():
        density_map = {}

        aggregate = []
        for word in value:
            if not isinstance(word, basestring):
                continue
            reformatted = word.lower().replace("+", "-")

            if reformatted not in filter_words+colors+numbers:
                aggregate.append(reformatted)

        words = aggregate

        for word in words:
            density_map[word] = model.neighbor_density_cos(threshold, word, words)

        output_path = os.path.join("data", "output",
                                   path, "semgraphs",
                                   "cosine_{}".format(threshold))

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        output_file = os.path.join(output_path, "semgraph_{}".format(key))


        with open(output_file, "wb") as output:
            json.dump(density_map, output, indent=4, sort_keys=True)



def generate_euclid_graph(model, path, wordmap, threshold):
    worddensity = {}

    for key, value in wordmap.items():
        if key not in worddensity:
            worddensity[key] = {}
        aggregate = []
        for word in value:
            #aggregate += filter(lambda x: x not in filter_words+colors+numbers, word.lower().split("+"))

            reformatted = word.lower().replace("+", "-")
            if reformatted not in filter_words+colors+numbers:
                aggregate.append(reformatted)
        aggregate = filter_plurals(aggregate)
        words = set(aggregate)

        for word in words:
            worddensity[key][word] = model.neighbor_density_euclid(threshold, word, words)

        output_path = os.path.join("data", "output",
                                   path, "semgraphs",
                                   "euclid_{}".format(threshold))

        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        output_file = os.path.join(output_path, "semgraph_{}".format(key))

        with open(output_file, "wb") as output:
            json.dump(worddensity[key], output, indent=4, sort_keys=True)


def create_numpy_from_glove(input, dict_out, vec_out):
    """

    This may take just short of forever, and potentially
    blow up your computer. Use with caution, don't be a hero.

    :param input: path to glove pretrained vectors
    :param dict_out: output path of dictionary
    :param vec_out: output path of numpy array
    :return:
    """
    glove = load_glove_pretrain(input)
    glove.save_model(dict_out, vec_out)


if __name__ == "__main__":

    path = sys.argv[1]


    # glove = GloVe("data/model/dict_glove_42b_300", "data/model/vectors_glove_42b_300.npy")
    #
    # wordmap = {}
    #
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         seedlings.load_basic_levels(os.path.join(root, file), wordmap)
    #
    #
    #
    # glove.graph_cosine_range("test", wordmap=wordmap,
    #                          start=0.5, end=0.51, step=0.1)

    # with open("data/density_euc_1/density6", "rU") as input:
    #     month6 = json.load(input)
    #
    # top_10 = top_n_words(month6, 10)

    wordbank_english = WordBank(input="data/wb_cdi/wb_eng.csv")

    graph_path = "data/output/english_wordbank/semgraphs/cosine_0.4/semgraph_wb_eng"

    graph = SemanticGraph(source="WordBank", sim_func="cos",
                        thresh=0.45, path=graph_path, wb=wordbank_english)

    top_n = graph.top_n_dense(10)