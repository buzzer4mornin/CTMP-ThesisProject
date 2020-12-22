import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
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
    print("Movie descriptions size:", len(desc))

    #TODO: Implement index-frequency
    # check [https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html]
    #for d in desc:
    #    print(word_tokenize(d))
    #ss = [word_tokenize(i) for i in desc]  # 2nd: list(map(word_tokenize, train2))
    #print(ss)
    #from gensim import corpora
    #dictionary = corpora.Dictionary(texts)

    stop_words = set(stopwords.words('english'))
    all_desc = ' '.join(desc).lower()
    all_word_tokens = word_tokenize(all_desc) if regex is None else RegexpTokenizer(regex).tokenize(all_desc)
    all_word_tokens = [w for w in all_word_tokens if not w in stop_words]


    #TODO: Start without Lemmatization/Stemming
    #TODO: Sort Vocabulary
    #Create Vocabulary textfile
    vocab = set(list(all_word_tokens))
    with open("vocabulary.txt", 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word)
            f.write("\n")

    # 20-60 seconds total execution time
    print("word_tokens size:", len(all_word_tokens))
    print("vocab size:", len(vocab))
    print("word/voc shrinkage:", round(len(all_word_tokens)/len(vocab)))
    print("NLTK time: {:.1f} seconds".format(time.time() - start))

    return all_word_tokens, vocab



"""
NLTK version
"""
my_text = train.copy()
my_text = my_text[200:1000] # subselecting only [:2] descriptions
_example = ["This is a foo_bar sentence. #$^&*@hello_world.123", "Aydin's solution, agil's, is his number of 100s 70s, 50 "]
#TODO: combine both regex
reg = r'[^\W_]+|[^\W_\s]+' # handles underscores, but cant handle less than 3
reg = r'\w{3,}'            # handles less than 3, but cant handle underscore
#_, vocab = nltk_token_vocab(my_text, regex=reg)




"""
Compare regular expression formulas
"""
#compare_regex("vocab_regex1.txt", "vocab_regex2.txt")




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