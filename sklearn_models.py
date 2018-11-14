import pickle
import os
import time

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

MODEL_DIR = 'models/sklearn'


def build_SKM(model_type=None, max_features=None, selectK=None, params={}):
    if not model_type:
        raise NameError('model_type is not defined')
    # Multinomial Naive Bayes
    if model_type is 'MNB':
        alpha = params.get('alpha', .01)
        pipe = Pipeline([('MNB', MultinomialNB(alpha=alpha))])

    # Suport Vector Machine (Linear Kernel)
    if model_type is 'SVC':
        pipe = Pipeline([('SVC', SVC(kernel='linear'))])

    # Extra Tree Classifier
    if model_type is 'ETC':
        n_estimators = params.get('n_estimators', 200)
        min_samples_split = params.get('min_samples_split', 50)
        pipe = Pipeline([
            ('ETC', ExtraTreesClassifier(n_estimators=n_estimators,
                                         min_samples_split=min_samples_split))
        ])
    # Random Forest Classifier
    if model_type is 'RFC':
        n_estimators = params.get('n_estimators', 200)
        min_samples_split = params.get('min_samples_split', 50)
        pipe = Pipeline([
            ('RFC', RandomForestClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split))
        ])
    # Feature Selection
    if selectK:
        pipe.steps.insert(0, ('fselect', SelectKBest(chi2, k=selectK)))

    # Tfidf Vectorizer
    pipe.steps.insert(0, ('TfidfVec', TfidfVectorizer(max_features=max_features)))
    return pipe


def save_model(model, model_name):
    timestamp = int(time.time())
    filename = f'{model_name}_{timestamp}'
    filepath = os.path.join(MODEL_DIR, filename)
    pickle.dump(model, open(filepath, 'wb'))


def load_model(model_name):
    filepath = os.path.join(MODEL_DIR, model_name)
    return pickle.load(open(filepath, 'rb'))


# def build_MNB(max_features=20000, bestK=10000, alpha=.01):
#     return Pipeline([
#         ('TfidfVec', TfidfVectorizer(max_features=max_features)),
#         ('fselect', SelectKBest(chi2, bestK)),
#         ('MNB', MultinomialNB(alpha=alpha))
#     ])


# def build_SVC():
#     return Pipeline([
#         ('TfidfVec', TfidfVectorizer(max_features=20000)),
#         ('fselect', SelectKBest(chi2, k=10000)),
#         ('SVC', SVC(kernel='linear'))
#     ])


# def build_ETC():
#     return Pipeline([
#         ('TfidfVec', TfidfVectorizer(max_features=20000)),
#         ('fselect', SelectKBest(chi2, k=10000)),
#         ('ETC', ExtraTreesClassifier(n_estimators=200, min_samples_split=50))
#     ])


# def build_RFC():
#     return Pipeline([
#         ('TfidfVec', TfidfVectorizer(max_features=20000)),
#         ('fselect', SelectKBest(chi2, k=10000)),
#         ('ETC', RandomForestClassifier(n_estimators=200, min_samples_split=50))
#     ])
