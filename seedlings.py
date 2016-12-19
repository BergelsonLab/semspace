import csv
import sys
import os
import pandas


def load_basic_levels(path, wordmap):
    """
    wordmap is a dict of month to list-of-words mappings

    :param path:
    :param wordmap:
    :return:
    """
    key = int(os.path.basename(path)[:2])
    words = pandas.read_csv(path)["word"]
    if key not in wordmap:
        wordmap[key] = set(words.tolist())

def load_seedlings(path):
    wordmap = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            load_basic_levels(os.path.join(root, file), wordmap)

if __name__ == "__main__":

    start_dir = sys.argv[1]

    seedlings_wordmap = load_seedlings(start_dir)

    print
