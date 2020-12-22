import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
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


def nltk_token_vocab(corpus):
    """
    corpus = ''.join(corpus)
    For now, -*- type(corpus) is str -*- (single long text, i.e combination of all movie description sentences)
    """
    start = time.time()
    corpus = ''.join(corpus)
    if type(corpus) is str: # SURE it is str for now
        word_tokens = word_tokenize(corpus)
    else:
        word_tokens = [word_tokenize(i) for i in corpus] #2nd: list(map(word_tokenize, train2))

    #TODO: Sort Vocabulary
    #Create Vocabulary textfile
    vocab = set(list(word_tokens))
    with open("vocabulary.txt", 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(word)
            f.write("\n")

    # 20-60 seconds total execution time
    print("Corpus size:", len(corpus))
    print("word_tokens size:", len(word_tokens))
    print("vocab size:", len(vocab))
    print("word/voc shrinkage:", int(len(word_tokens)/len(vocab)))
    print("NLTK time: {:.1f} seconds".format(time.time() - start))

    return word_tokens, vocab


#NLTK version
my_text = train.copy()
#for i in range(2):
#    my_text += train

_, vocab = nltk_token_vocab(my_text)



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