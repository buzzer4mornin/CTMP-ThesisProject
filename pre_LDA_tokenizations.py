import time
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from nltk.tokenize import word_tokenize
import numpy as np

# grab a sample data set
dataset = fetch_20newsgroups(shuffle=True, remove=('headers', 'footers', 'quotes'))
train, test = dataset.data, dataset.data


"""
[M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

[M]     - number of unique terms in document. 
[count] - how many times each term appeared in the document.
[term]  - integer which indexes the term; it is not a string.

Assuming Movie Descriptions will be stored as string in Oracle Database



"""



def nltk_tokenize(corpus):
    """
    corpus = ''.join(my_text)
    For now, -*- type(corpus) is str -*- (single long text, i.e combination of all movie description sentences)
    """
    start = time.time()
    print("Corpus size:", len(corpus))
    if type(corpus) is str:
        word_tokens = word_tokenize(corpus)
    else:
        word_tokens = [word_tokenize(i) for i in corpus] #2nd: list(map(word_tokenize, train2))
    voc = set(list(word_tokens))
    print("word_tokens size:", len(word_tokens))
    print("vocab size:", len(voc))
    print("word/voc shrinkage", int(len(word_tokens)/len(voc)))
    print("NLTK time:", time.time() - start)
    exit()
    # 20-60 seconds interval


#TODO: NLTK vs Spacy
sp = spacy.load('en_core_web_sm')

my_text = train.copy()
for i in range(2):
    my_text += train

my_text = ''.join(my_text)
nltk_tokenize(my_text)







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