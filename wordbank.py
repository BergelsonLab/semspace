import sys
import pandas

class WordBank(object):

    def __init__(self, input):
        self.data = self.load_file(input)

    def load_file(self, input):
        self.data = pandas.read_csv(input)

    def top_n_month(self, month, n):
        result = self.data.sort_values(by=month, ascending=False)
        return result[:n]


# def top_n_month(data, month, n):
#     result = data.sort_values(by=month, ascending=False)
#     return result[:n]



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