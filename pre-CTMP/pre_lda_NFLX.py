import os
import time
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.datasets import fetch_20newsgroups
import re

# Change to Current File Directory
os.chdir(os.path.dirname(__file__))

"""
[M] [word_1]:[count] [word_2]:[count] ...  [word_N]:[count]
[M]     - number of unique words in plot
[word]  - integer which is index of the word in vocabulary
[count] - how many times each word appeared in the plot

"""


def get_vocabulary(plots):
    """ Create and return vocabulary out of descriptions """
    if os.path.exists("vocab_NFLX.txt"):
        os.remove("vocab_NFLX.txt")
    start = time.time()
    stop_words = set(stopwords.words('english'))

    # Get all terms in all plots. Deduct stop-words from it.
    all_plots = ' '.join(plots).lower()
    all_plots = all_plots.replace("&amp;quot;", "")  # remove amp;quot
    all_plots = re.sub(r"\S*\d\S*", "", all_plots).strip()  # remove words with numbers
    all_terms = RegexpTokenizer(r'\w{3,}').tokenize(all_plots)  # tokenize -> then remove words of length < 3
    all_terms = [w for w in all_terms if w not in stop_words and "_" not in w]  # remove words with underscore

    # Create Vocabulary textfile
    vocab = sorted(set(list(all_terms)))
    with open("vocab_NFLX.txt", 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word + "\n")
    print("Total # of plots [or # of movies] :", len(plots))
    print("Total # of terms in all plots: ", len(all_terms))
    print("Vocabulary size of all plots: {}".format(len(vocab)))
    print("Terms/Vocab shrinkage: {:.1f}".format(len(all_terms) / len(vocab)))
    print('Execution time: {:.2f} seconds'.format(time.time() - start))
    print('-*-*-* Successfully Created "vocab_NFLX.txt" *-*-*- \n')
    return vocab


def get_input_movies(vocab, plots):
    """ Create input text file for LDA """
    if os.path.exists("movies_NFLX.txt"):
        os.remove("movies_NFLX.txt")
    # TODO write separate def for below section??
    start = time.time()
    term_vs_index = {v_term: v_index for v_term, v_index in zip(vocab, range(len(vocab)))}
    stop_words = set(stopwords.words('english'))
    for plt in plots:
        plt = plt.lower()
        try:
            plt = plt.replace("&amp;quot;", "")  # remove amp;quot
            plt = re.sub(r"\S*\d\S*", "", plt).strip()  # remove words with numbers
        except:
            print("ERROR in handling [&amp;quot] or [numbers]")
        terms = RegexpTokenizer(r'\w{3,}').tokenize(plt)  # tokenize -> then remove words of length < 3
        terms = [t for t in terms if t not in stop_words and "_" not in t]  # remove words with underscore
        term_counts = {}
        for t in terms:
            try:
                term_counts[term_vs_index[t]] += 1
            except KeyError:
                term_counts[term_vs_index[t]] = 1
        unique_terms = len(term_counts.keys())
        if unique_terms == 0:
            continue
        term_counts = str(term_counts).replace("{", "").replace("}", "").replace(" ", "").replace(",", " ")
        with open("movies_NFLX.txt", 'a', encoding='utf-8') as f:
            f.write(str(unique_terms) + " " + term_counts + "\n")
    end = time.time()
    print("Execution time: {:.2f} seconds".format(end - start))
    print('-*-*-* Successfully Created "movies_NFLX.txt" *-*-*-')


if __name__ == '__main__':
    movie_df = pd.read_pickle("../db-files/processed-files/df_movie_NFLX_UPDATED")
    movie_plt = movie_df["MOVIEPLOT"].tolist()

    # Run Experiment
    vocab = get_vocabulary(movie_plt)
    get_input_movies(vocab, movie_plt)
