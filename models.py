import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


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
