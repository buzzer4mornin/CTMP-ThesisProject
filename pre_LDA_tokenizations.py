import time
import os
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from gensim import corpora
import numpy as np
sp = spacy.load('en_core_web_sm')


# grab a sample data set
dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
train, test = dataset.data, dataset.data

"""
[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

[M]     - number of unique terms in document. 
[count] - how many times each term appeared in the document.
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


def nltk_token_vocab(desc, regex=None):
    start = time.time()
    stop_words = set(stopwords.words('english'))
    print("Movie descriptions size:", len(desc))

    # Clean all_word_tokes, leave only vocab
    all_desc = ' '.join(desc).lower()
    all_word_tokens = word_tokenize(all_desc) if regex is None else RegexpTokenizer(regex).tokenize(all_desc)
    all_word_tokens = [w for w in all_word_tokens if not w in stop_words]

    # Create Vocabulary textfile
    # DONE: regex draft + sorted vocabulary
    # TODO: try optional Lemmatization/Stemming
    vocab = sorted(set(list(all_word_tokens)))
    with open("vocabulary.txt", 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word)
            f.write("\n")

    # 20-60 seconds total execution time
    print("word_tokens size:", len(all_word_tokens))
    print("vocab size:", len(vocab))
    print("word/voc shrinkage: {:.1f}".format(len(all_word_tokens)/len(vocab)))
    print("NLTK time: {:.2f} seconds".format(time.time() - start))

    # Separate dictionaries for each descriptions
    # check [https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html]
    # TODO: Optimize index searching part etc
    if os.path.exists("result.txt"):
        os.remove("result.txt")
    for d in desc:
        d = d.lower()
        words = word_tokenize(d) if regex is None else RegexpTokenizer(regex).tokenize(d)
        words = [w for w in words if not w in stop_words]
        dict = {}
        for i in words:
            try:
                dict[vocab.index(i)] += 1
            except KeyError:
                dict[vocab.index(i)] = 1
        unique = str(len(dict.keys()))
        if unique == "0":
            continue
        map_counts = str(dict).replace("{", "").replace("}", "").replace(" ", "").replace(",", " ")
        with open("result.txt", 'a', encoding='utf-8') as f:
            f.write(unique)
            f.write(" ")
            f.write(map_counts)
            f.write("\n")

    return all_word_tokens, vocab



# ========================================== NLTK version ==============================================================
my_text = train.copy()
for i in range(2):
    my_text += train
my_text = my_text[0:200] # subselecting only [:2] descriptions
_example1 = ["This is a foo_bar this sentence sentence. But also this sentence #$^&*@hello_world.123", "Aydin's solution, aydin's solution aydin , is his number of 100s 70s, 50 "]
_example2 = ["*Contains spoilers due to me having to describe some film techniques, so read at your own risk!*<br /><br />I loved this film. The use of tinting in some of the scenes makes it seem like an old photograph come to life. I also enjoyed the projection of people on a back screen. For instance, in one scene, Leopold calls his wife and she is projected behind him rather than in a typical split screen. Her face is huge in the back and Leo's is in the foreground.<br /><br />One of the best uses of this is when the young boys kill the Ravensteins on the train, a scene shot in an almost political poster style, with facial close ups. It reminded me of Battleship Potemkin, that intense constant style coupled with the spray of red to convey tons of horror without much gore. Same with the scene when Katharina finds her father dead in the bathtub...you can only see the red water on the side. It is one of the things I love about Von Trier, his understatement of horror, which ends up making it all the more creepy.<br /><br />The use of text in the film was unique, like when Leo's character is pushed by the word, 'Werewolf.' I have never seen anything like that in a film.<br /><br />The use of black comedy in this film was well done. Ernst-Hugo Järegård is great as Leo's uncle. It brings up the snickers I got from his role in the Kingdom (Riget.) This humor makes the plotline of absurd anal retentiveness of train conductors against the terrible backdrop of WW2 and all the chaos, easier to take. It reminds me of Riget in the way the hospital administrator is trying to maintain a normalcy at the end of part one when everything is going crazy. It shows that some people are truly oblivious to the awful things happening around them. Yet some people, like Leo, are tuned in, but do nothing positive about it.<br /><br />The voice over, done expertly well by Max von Sydow, is amusing too. It draws you into the story and makes you jump into Leo's head, which at times is a scary place to be.<br /><br />The movie brings up the point that one is a coward if they don't choose a side. I see the same idea used in Dancer in the Dark, where Bjork's character doesn't speak up for herself and ends up being her own destruction. Actually, at one time, Von Trier seemed anti-woman to me, by making Breaking the Waves and Dancer, but now I know his male characters don't fare well either! I found myself at the same place during the end of Dancer, when you seriously want the main character to rethink their actions, but of course, they never do!"]
_example3 = ["Place is near restaurant, near the corner"]

#TODO: combine both regex
reg1 = r'[^\W_]+|[^\W_\s]+' # handles underscores, but cant handle less than 3
reg2 = r'\w{3,}'            # handles less than 3, but cant handle underscore
_, vocab = nltk_token_vocab(my_text, regex=reg2)


#Compare regular expression formulas
#compare_regex("vocab_regex1.txt", "vocab_regex2.txt")



# ========================================= SpaCy version ==============================================================
#TODO: SpaCy version
'''start = time.time()
train2 = train.copy()
spacy_nlp = spacy.load('en_core_web_sm')
tokenized_sents = [[token.text for token in spacy_nlp(article)] for article in train2]
print("spacy time:", time.time() - start)'''
'''train = ["My name is name", "Your name is mine"]

# vectorizer the features
tf_vectorizer = CountVectorizer(max_features=25)
X_train = tf_vectorizer.fit_transform(train)
print(X_train)

# train the model
lda = LatentDirichletAllocation(n_topics=5)
lda.fit(X_train)

# predict topics for test data
# unnormalized doc-topic distribution
X_test = tf_vectorizer.transform(test)
doc_topic_dist_unnormalized = np.matrix(lda.transform(X_test))

# normalize the distribution (only needed if you want to work with the probabilities)
doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)'''