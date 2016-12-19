import words
import json
import os
import csv



def rank_density(input_path="", output_path=""):
    if input_path and output_path:
        input_folder = input_path
        output_folder = output_path

    elif input_path and not output_path:
        input_folder = input_path
        output_folder = os.path.join("data/ranked_out", os.path.basename(input_folder))
    else:
        input_folder = "data/semgraphs"
        output_folder = "data/ranked_out"

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.startswith("."):
                with open(os.path.join(root, file), "rU") as input:
                    semgraph = json.load(input)
                    top_n = words.top_n_words(semgraph, 50)
                    just_words = []
                    for entry in top_n:
                        just_words.append((entry[0], len(entry[1])))

                    if not input_path:
                        final_out_folder = root.replace(input_folder, output_folder)
                    else:
                        final_out_folder = os.path.join(output_folder, os.path.basename(root))

                    if not os.path.isdir(final_out_folder):
                        os.makedirs(final_out_folder)

                    final_out = os.path.join(final_out_folder, file+".csv")
                    with open(final_out, "wb") as output:
                        writer = csv.writer(output)
                        for word in just_words:
                            writer.writerow([word[0], word[1]])