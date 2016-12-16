import csv
import sys
import os


def load_basic_levels(path, wordmap):
    """
    wordmap is a dict of month to list-of-words mappings

    :param path:
    :param wordmap:
    :return:
    """
    with open(path, "rU") as input:
        key = int(os.path.basename(path)[:2])
        reader = csv.reader(input)
        reader.next()

        if key not in wordmap:
            wordmap[key] = []

        for row in reader:
            wordmap[key].append(row[1])


if __name__ == "__main__":

    start_dir = sys.argv[1]

    wordmap = {}

    for root, dirs, files in os.walk(start_dir):
        for file in files:
            load_basic_levels(os.path.join(root, file), wordmap)

    print
