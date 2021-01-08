import os
import time
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.datasets import fetch_20newsgroups

# Change to Current File Directory
os.chdir(os.path.dirname(__file__))

# TODO: rename word -> term in code
# TODO: update row counts below
"""
[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]
[M]     - number of unique terms in plot. 
[count] - how many times each term appeared in the plot.
[term]  - integer which indexes the term; it is not a string.

DB Row Counts: [MOVIES - 27,278] [USERS - 138,493] [RATINGS - 20,000,263] 
"""


def compare_regex(reg_file1, reg_file2):
    with open(reg_file1, "r") as f1, open(reg_file2, "r") as f2:
        voc1 = f1.readlines()
        voc2 = f2.readlines()
    non = []
    for word in voc2:
        if word not in voc1: non.append(word)
    with open("_difference.txt", 'w', encoding='utf-8') as f:
        for word in non:
            f.write(word)


def get_vocabulary(plots, regex=None):
    """ Create and return vocabulary out of descriptions """
    if os.path.exists("vocab.txt"):
        os.remove("vocab.txt")
    start = time.time()
    stop_words = set(stopwords.words('english'))
    print("Total # of plots [or # of movies] :", len(plots))

    # Get all terms in all plots. Deduct stop-words from it.
    all_plots = ' '.join(plots).lower()
    all_terms = word_tokenize(all_plots) if regex is None else RegexpTokenizer(regex).tokenize(all_plots)
    all_terms = [w for w in all_terms if w not in stop_words]
    print("Total # of terms in all plots: ", len(all_terms))

    # Create Vocabulary textfile
    # TODO: try optional Lemmatization/Stemming
    vocab = sorted(set(list(all_terms)))
    print("Vocabulary size of all plots: {}".format(len(vocab)))
    ##### weird = lambda s: any(i.isdigit() or i == "_" for i in s)
    with open("vocab.txt", 'w', encoding='utf-8') as f:
        for word in vocab:
        #####    if not weird(word):
            f.write(word + "\n")
    print("Terms/Vocab shrinkage: {:.1f}".format(len(all_terms) / len(vocab)))
    print('Execution time for "vocab.txt": {:.2f} seconds'.format(time.time() - start))
    print("-*-*-*-")
    return vocab


def get_input(vocab, plots, regex=None):
    """ Create input text file for LDA """
    if os.path.exists("input.txt"):
        os.remove("input.txt")
    # TODO write separate def for below section??
    start = time.time()
    term_vs_index = {v_term: v_index for v_term, v_index in zip(vocab, range(len(vocab)))}
    stop_words = set(stopwords.words('english'))
    for plt in plots:
        plt = plt.lower()
        terms = word_tokenize(plt) if regex is None else RegexpTokenizer(regex).tokenize(plt)
        terms = [t for t in terms if t not in stop_words]
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
        with open("input.txt", 'a', encoding='utf-8') as f:
            f.write(str(unique_terms) + " " + term_counts + "\n")
    end = time.time()
    print('Execution time for "input.txt": {:.2f} seconds'.format(end - start))
    print("-*-*-*-")


if __name__ == '__main__':
    opt = 1
    if opt == 1:
        # Incorporate df_movie [getting movie plots from DB]
        movie_df = pd.read_pickle("../db-files/df_movie")
        movie_plt = movie_df["MOVIEPLOT"].tolist()

        # Run Experiment
        reg1 = r'[^\W_]+|[^\W_\s]+'  # handles underscores, but cant handle less than 3
        reg2 = r'\w{3,}'  # ^[0-9]+$ handles less than 3, but cant handle underscore
        vocab = get_vocabulary(movie_plt, regex=reg2)
        get_input(vocab, movie_plt, regex=reg2)

        # Compare RegEx formulas
        # compare_regex("vocab_regex1.txt", "vocab_regex2.txt")

    else:
        # Grab a sample Dataset
        dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
        train, test = dataset.data, dataset.data

        # Subselect Dataset
        my_text = train.copy()
        for i in range(2):
            my_text += train
        my_text = my_text[0:10000]  # [16 seconds = 30,000 descriptions/100,000 vocab]
        _example1 = ["This is a foo_bar this sentence sentence. But also this sentence #$^&*@hello_world.123",
                     "car's solution, red's solution car , is his number of 100s 70s, 50 "]
        _example2 = [
            "*Contains spoilers due to me having to describe some film techniques, so read at your own risk!*<br /><br />I loved this film. The use of tinting in some of the scenes makes it seem like an old photograph come to life. I also enjoyed the projection of people on a back screen. For instance, in one scene, Leopold calls his wife and she is projected behind him rather than in a typical split screen. Her face is huge in the back and Leo's is in the foreground.<br /><br />One of the best uses of this is when the young boys kill the Ravensteins on the train, a scene shot in an almost political poster style, with facial close ups. It reminded me of Battleship Potemkin, that intense constant style coupled with the spray of red to convey tons of horror without much gore. Same with the scene when Katharina finds her father dead in the bathtub...you can only see the red water on the side. It is one of the things I love about Von Trier, his understatement of horror, which ends up making it all the more creepy.<br /><br />The use of text in the film was unique, like when Leo's character is pushed by the word, 'Werewolf.' I have never seen anything like that in a film.<br /><br />The use of black comedy in this film was well done. Ernst-Hugo Järegård is great as Leo's uncle. It brings up the snickers I got from his role in the Kingdom (Riget.) This humor makes the plotline of absurd anal retentiveness of train conductors against the terrible backdrop of WW2 and all the chaos, easier to take. It reminds me of Riget in the way the hospital administrator is trying to maintain a normalcy at the end of part one when everything is going crazy. It shows that some people are truly oblivious to the awful things happening around them. Yet some people, like Leo, are tuned in, but do nothing positive about it.<br /><br />The voice over, done expertly well by Max von Sydow, is amusing too. It draws you into the story and makes you jump into Leo's head, which at times is a scary place to be.<br /><br />The movie brings up the point that one is a coward if they don't choose a side. I see the same idea used in Dancer in the Dark, where Bjork's character doesn't speak up for herself and ends up being her own destruction. Actually, at one time, Von Trier seemed anti-woman to me, by making Breaking the Waves and Dancer, but now I know his male characters don't fare well either! I found myself at the same place during the end of Dancer, when you seriously want the main character to rethink their actions, but of course, they never do!"]
        _example3 = ["Place is near restaurant, near the corner"]

        # TODO: combine both regex
        reg1 = r'[^\W_]+|[^\W_\s]+'  # handles underscores, but cant handle less than 3
        reg2 = r'\w{3,} +|^\d+$ '  # ^[0-9]+$ handles less than 3, but cant handle underscore
        my_vocabulary = get_vocabulary(my_text, regex=reg2)
        get_input(my_vocabulary, my_text, regex=reg2)

        # Compare RegEx formulas
        # compare_regex("vocab_regex1.txt", "vocab_regex2.txt")
