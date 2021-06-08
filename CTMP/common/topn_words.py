#! /usr/bin/python
import sys
import math
import numpy as np
import utilities
# ------------ RUN in terminal ------------
# --> python ./model/topn_words.py original
# --> python ./model/topn_words.py reduced
# --> python ./model/topn_words.py diminished

# blei topics: http://www.cs.columbia.edu/~blei/lda-c/ap-topics.pdf

# TODO: move to ./common directory

def print_topics(vocab_file, nwords, result_file):
    with open(vocab_file, 'r') as f:
        vocab = f.readlines()

    vocab = list(map(lambda x: x.strip(), vocab))
    vocab_index = {i: w for i, w in zip(range(len(vocab)), vocab)}

    with open(nwords, "r") as n:
        lines = n.readlines()
        for l in lines:
            l = l.split()
            converts = list(map(lambda x: vocab_index[int(x)], l))
            converts = " ".join(converts)
            with open(result_file, "a") as r:
                r.write(converts + "\n")


if __name__ == '__main__':
    # beta = np.load("./beta.npy")
    # list_tops = utilities.list_top(beta, 20)
    #
    # def write_topic_top(list_tops, file_name):
    #     num_topics = len(list_tops)
    #     tops = len(list_tops[0])
    #     f = open(file_name, 'w')
    #     for k in range(num_topics):
    #         for j in range(tops - 1):
    #             f.write('%d ' % (list_tops[k][j]))
    #         f.write('%d\n' % (list_tops[k][tops - 1]))
    #     f.close()
    #
    #
    # write_topic_top(list_tops, "./list_tops.txt")

    list_tops = "./list_tops.txt"
    vocab_file = "./vocab_NFLX.txt"
    result_file = "./topn_output.txt"
    print_topics(vocab_file, list_tops, result_file)

