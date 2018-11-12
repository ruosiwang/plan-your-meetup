import re
import os
import numpy as np
from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


GLOVE_DIR = 'data/glove.6B'

lemmatizer = WordNetLemmatizer()
junk = ['href', 'http', 'https', 'span', 'amp', 'nbsp', 'org', 'com', 'www', 'facebook', 'twitter', 'linkedin', 'meetup']
STOP_WORDS = stopwords.words("english") + junk


def clean_text(doc, lemmatize=False):
    """
    text_cleaning
    """
    # Remove HTML tags
    no_html = re.sub('<[^<]+?>', '', doc)
    # Remove non-letters
    words_only = re.sub("[^a-zA-Z]", " ", no_html)
    # lemmatize (only include words with more than 3 characters)
    words = []
    for word in words_only.lower().split():
        if len(word) >= 3 and (word not in STOP_WORDS):
            if lemmatize:
                words.append(lemmatizer.lemmatize(word))
            else:
                words.append(word)
    return " ".join(words)


def clean_texts(documents):
    return [clean_text(doc) for doc in documents]


def top_frequent_words(documents, n=100, ngram_range=(1, 1)):
    """
    return the top N of frequent words
    """
    countvec = CountVectorizer(ngram_range=ngram_range)
    counts = countvec.fit_transform(documents)
    counts = np.array(counts.sum(axis=0))[0]

    word_counts = list(zip(countvec.get_feature_names(), counts))
    return sorted(word_counts, key=lambda x: x[1], reverse=True)[:n]


def load_glove(embedding_dim=50):
    """
    build index mapping words in the embeddings set
    to their embedding vector (from GloVe)
    """
    print("Indexing GloVe word vectors...")
    embedding = {}
    file = os.path.join(GLOVE_DIR, f"glove.6B.{embedding_dim}d.txt")
    with open(file, "rb") as f:
        for line in f.readlines():
            values = line.decode().split()
            word = values[0]
            vec = np.asarray(values[1:], dtype=np.float)
            embedding[word] = vec
    return embedding


def get_embedding_dict(vocabulary, embedding_index):
    embeddings = {}
    for voc in embedding_index:
        lemma = lemmatizer.lemmatize(voc)
        if lemma in vocabulary:
            if not embeddings.get(lemma):
                embeddings[lemma] = [embedding_index[voc]]
            else:
                embeddings[lemma].append(embedding_index[voc])
    embeddings = {k: np.array(v).mean(axis=0) for k, v in embeddings.items()}
    return embeddings


def get_embedding_matrix(word_index, embedding_index,
                         MAX_NUM_WORDS, EMBEDDING_DIM):
    num_words = min(len(word_index), MAX_NUM_WORDS) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i < MAX_NUM_WORDS:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


class MeanEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embeddings, tfidfVec=None):
        self.embeddings = embeddings
        self.vocabulary = embeddings.keys()
        self.dim = len(list(embeddings.values())[0])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.embeddings[w] for w in words if w in self.embeddings] or
                    [np.zeros(self.dim)], axis=0) for words in X])


class TfidfEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embeddings, max_features=None):
        self.embeddings = embeddings
        self.dim = len(list(embeddings.values())[0])
        self.max_features = max_features
        self.weight = None

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer(max_features=self.max_features)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        return np.array([
                        np.mean([self.weight[w] * self.embeddings[w]
                                 for w in words.split() if w in self.embeddings] or
                                [np.zeros(self.dim)], axis=0)
                        for words in X
                        ])
