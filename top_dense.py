import words
import json
import os

import ipdb; ipdb.set_trace()


def rank(path=""):
    if path:
        input_folder = path
        output_folder = os.path.join("data/ranked_out", os.path.basename(input_folder))
    else:
        input_folder = "data/semgraphs"
        output_folder = "data/ranked_out"

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.startswith("."):
                with open(os.path.join(root, file), "rU") as input:
                    monthwords = json.load(input)
                    top_10 = words.top_n_words(monthwords, 50)
                    just_words = []
                    for entry in top_10:
                        just_words.append((entry[0], len(entry[1])))

                    if not path:
                        final_out_folder = root.replace(input_folder, output_folder)
                    else:
                        final_out_folder = output_folder
                        print final_out_folder

                    if not os.path.isdir(final_out_folder):
                        os.makedirs(final_out_folder)

                    with open(os.path.join(final_out_folder, file), "wb") as output:
                        for word in just_words:
                            output.write("{} {}\n".format(word[0], word[1]))


#rank()

rank(path="data/semgraphs/cosine_0.3")
