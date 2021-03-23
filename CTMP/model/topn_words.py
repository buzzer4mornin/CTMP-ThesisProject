#! /usr/bin/python
import sys
import math

# ------------ RUN in terminal ------------
# --> python ./model/topn_words.py original
# --> python ./model/topn_words.py reduced
# --> python ./model/topn_words.py diminished

# blei topics: http://www.cs.columbia.edu/~blei/lda-c/ap-topics.pdf


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
    if len(sys.argv) != 2 or sys.argv[1] not in ["original", "reduced", "diminished"]:
        print("WRONG USAGE! TRY --> python ./model/topn_words.py [original, reduced or diminished]")
        exit()

    which_size = sys.argv[1]
    vocab_file = "./input-data/vocab.txt" if which_size == "original" else "./input-data/vocab_REDUCED.txt" if which_size == "reduced" else "./input-data/vocab_DIMINISHED.txt"
    list_tops = "./output-data/list_tops.txt"
    result_file = "./output-data/topn_output.txt"
    print_topics(vocab_file, list_tops, result_file)

