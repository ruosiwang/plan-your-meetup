import re
import os
import numpy as np
# from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

GLOVE_DIR = 'data/glove.6B'

lemmatizer = WordNetLemmatizer()
junk = ['href', 'http', 'https', 'span', 'amp', 'nbsp', 'org', 'com', 'www', 'facebook', 'twitter', 'linkedin', 'meetup']
STOP_WORDS = stopwords.words("english") + junk


def clean_text(doc):
    """
    text_cleaning
    """
    # def clean_text(doc):
    cleaned = []

    # Remove HTML tags
    no_html = re.sub('<[^<]+?>', '', doc)

    # Remove non-letters
    words_only = re.sub("[^a-zA-Z]", " ", no_html)

    # lemmatize (only include words with more than 3 characters)
    for word in words_only.lower().split():
        if len(word) >= 3 and (word not in STOP_WORDS):
            cleaned.append(lemmatizer.lemmatize(word))
    return " ".join(cleaned)


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

# todo: some words in vocabulary are not included in the glove
# totoal words in embedding matrix (17715)
# totoal tokenized (20000)


def create_embedding_dict(vocabulary, embedding_dim):
    glove = load_glove(embedding_dim)

    embeddings = OrdereDict()
    for voc in glove:
        lemma = lemmatizer.lemmatize(voc)
        if lemma in vocabulary:
            if not embeddings.get(lemma):
                embeddings[lemma] = [glove[voc]]
            else:
                embeddings[lemma].append(glove[voc])
    embeddings = {k: np.array(v).mean(axis=0) for k, v in embeddings.items()}
    return embeddings


def embedding_embedding_dict(vocabulary, embedding_dim):
    embedding_index = create_embedding_dict(vocabulary, embedding_dim)
    return np.array(list(embedding_index.values()))
