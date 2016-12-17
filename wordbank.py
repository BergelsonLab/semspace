import sys
import os
import pandas

class WordBank(object):

    def __init__(self, input):
        self.input_file = input
        self.load_file(input)

    def load_file(self, input):
        self.data = pandas.read_csv(input)

    def top_n_month(self, month, n):
        result = self.data.sort_values(by=month, ascending=False)
        return result[:n]

    def wordmap(self):
        wordmap = {}
        data_set_name = os.path.basename(self.input_file).replace(".csv", "")
        wordmap[data_set_name] = self.data["definition"].values

        return wordmap



if __name__ == "__main__":

    input_file = sys.argv[1]

    wordbank = WordBank(input_file)

    data = pandas.read_csv(input_file)

    all_the_words = data["definition"].values



    top_16 = wordbank.top_n_month("16", 20).definition
    top_17 = wordbank.top_n_month("17", 20).definition
    top_18 = wordbank.top_n_month("18", 20).definition
    top_19 = wordbank.top_n_month("19", 20).definition
    top_20 = wordbank.top_n_month("20", 20).definition
    top_21 = wordbank.top_n_month("21", 20).definition
    top_22 = wordbank.top_n_month("22", 20).definition
    top_23 = wordbank.top_n_month("23", 20).definition


    print top_16.values

    print