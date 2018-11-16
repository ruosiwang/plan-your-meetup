import pickle
import os
import time
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


MODEL_DIR = 'models/sklearn'
DATA_DIR = 'data'


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
                                         min_samples_split=min_samples_split,
                                         n_jobs=-1))
        ])
    # Random Forest Classifier
    if model_type is 'RFC':
        n_estimators = params.get('n_estimators', 200)
        min_samples_split = params.get('min_samples_split', 50)
        pipe = Pipeline([
            ('RFC', RandomForestClassifier(n_estimators=n_estimators,
                                           min_samples_split=min_samples_split,
                                           n_jobs=-1))
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


def model_evaluations():
    models = [f for f in sorted(os.listdir(MODEL_DIR))
              if not f.startswith('.')]
    val_data = [f for f in sorted(os.listdir(DATA_DIR))
                if f.startswith('validation')]
    model_num = len(models) // len(val_data)

    f1_scores, acc_scores = [], []
    for i, val in enumerate(val_data):
        f1, acc = [], []
        [X_val, y_val] = pickle.load(
            open(os.path.join(DATA_DIR, val), 'rb'))

        for m in models[i:: model_num]:
            model = pickle.load(
                open(os.path.join(MODEL_DIR, m), 'rb'))

            y_pred = model.predict(X_val)
            f1.append(f1_score(y_pred, y_val, average='weighted'))
            acc.append(accuracy_score(y_pred, y_val))

        f1_scores.append(f1)
        acc_scores.append(acc)
    return f1_scores, acc_scores


def most_important_features(model, topN):
    # features
    tfidf = model.steps[0][0]
    features = model.named_steps[tfidf].get_feature_names()

    if len(model.steps) == 3:
        kbest = model.steps[1][0]
        mask = model.named_steps[kbest].get_support()
        features = [f for f, m in zip(features, mask) if m]

    # classifier coefs
    clf = model.steps[-1][0]
    coef = model.named_steps[clf].coef_

    top_features = []
    for ctg in range(len(coef)):
        top_index = np.argsort(coef[ctg])[-topN:]
        top_features.append([features[i] for i in top_index])

    return top_features


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
