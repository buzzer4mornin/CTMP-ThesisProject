#! /usr/bin/python
import sys
import math
import numpy as np
import os

os.chdir('../common/')

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
    os.chdir('../experimentation/sparsity&topn')
    list_tops = "./list_tops.txt"
    vocab_file = "vocab_NFLX.txt"
    result_file = "topn_output.txt"
    print_topics(vocab_file, list_tops, result_file)

