import array
import sys
import json
import operator
import os
import fileinput
import numpy as np

import loadwords


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
    def __init__(self):
        self.dim = None
        self.dict = None
        self.vectors = None

    def save_model(self, dict, vectors):
        np.save(vectors, self.vectors)
        with open(dict, "wb") as output:
            json.dump(self.dict, output)

    def load_model(self, dict, vectors):
        self.vectors = np.load(vectors)
        with open(dict, "rU") as input:
            self.dict = json.load(input)

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

    # with open(path, 'rU') as input:
    #     for i, line in enumerate(input):
    #         tokens = line.split()
    #
    #         word = tokens[0]
    #         entries = tokens[1:]
    #
    #         dct[word] = i
    #         vectors.extend(float(x) for x in entries)
    #         # if i > 10000:
    #         #     break


    i = 0
    for line in fileinput.input([path]):
        tokens = line.split()

        word = tokens[0]
        entries = tokens[1:]

        dct[word] = i
        vectors.extend(float(x) for x in entries)
        i+=1



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



def generate_cosine_graph(wordmap, threshold):
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
            worddensity[key][word] = glove.neighbor_density_cos(threshold, word, words)

        if not os.path.isdir("data/semgraphs/cosine_{}".format(threshold)):
            os.makedirs("data/semgraphs/cosine_{}".format(threshold))

        with open("data/semgraphs/cosine_{}/semgraph_{}".format(threshold, key), "wb") as output:
            json.dump(worddensity[key], output, indent=4, sort_keys=True)


def generate_euclid_graph(wordmap, threshold):
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
            worddensity[key][word] = glove.neighbor_density_euclid(threshold, word, words)

        if not os.path.isdir("data/semgraphs/euclid_{}".format(threshold)):
            os.makedirs("data/semgraphs/euclid_{}".format(threshold))

        with open("data/semgraphs/euclid_{}/semgraph_{}".format(threshold, key), "wb") as output:
            json.dump(worddensity[key], output, indent=4, sort_keys=True)


if __name__ == "__main__":

    path = sys.argv[1]




    #
    # path = "data/model/glove.42B.300d.txt"
    # glove = load_glove_pretrain(path)
    #
    # glove.save_model("dict_glove_42b_300", "vectors_glove_42b_300")







    glove = GloVe()
    glove.load_model("data/model/dict_glove_42b_300", "data/model/vectors_glove_42b_300.npy")

    wordmap = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            loadwords.load_basic_levels(os.path.join(root, file), wordmap)



    generate_cosine_graph(wordmap, 0.62)

    #generate_euclid_graph(wordmap, 0.9)














    # with open("data/density_euc_1/density6", "rU") as input:
    #     month6 = json.load(input)
    #
    # top_10 = top_n_words(month6, 10)
