import argparse
import numpy as np
import sys

def generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab.txt', type=str)
    parser.add_argument('--vectors_file', default='vectors.txt', type=str)
    args = parser.parse_args()

    with open(args.vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(args.vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)

def generate_default():
    with open("vocab.txt", 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open("vectors.txt", 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance_original(W, vocab, ivocab, input_term):
    N = 30
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :]
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        index = vocab[term]
        dist[index] = -np.Inf


    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))

def distance(W, vocab, ivocab, word1, word2):
    if word1 not in vocab:
        print('Word: %s  Out of dictionary!\n' % word1)
        return
    else:
        w1_vector = W[vocab[word1], :]
    if word2 not in vocab:
        print('Word: %s  Out of dictionary!\n' % word2)
        return
    else:
        w2_vector = W[vocab[word2], :]

    w1_vec_norm = np.zeros(w1_vector.shape)
    d1 = (np.sum(w1_vector ** 2,) ** (0.5))
    w1_vec_norm = (w1_vector.T / d1).T

    w2_vec_norm = np.zeros(w2_vector.shape)
    d2 = (np.sum(w2_vector ** 2, ) ** (0.5))
    w2_vec_norm = (w2_vector.T / d2).T

    dist = np.dot(w1_vec_norm.T, w2_vec_norm.T)

    return dist


if __name__ == "__main__":
    N = 100          # number of closest words that will be shown
    W, vocab, ivocab = generate()
    while True:
        input_term = raw_input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            split_terms = input_term.split()
            #distance(W, vocab, ivocab, input_term)
            distance2(W, vocab, ivocab, split_terms[0], split_terms[1])
