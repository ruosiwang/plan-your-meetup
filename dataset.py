import helper as h

import os
import gzip
import random
import pickle
import ujson as json
import numpy as np
from collections import Counter
from itertools import islice, chain

from sklearn.feature_extraction.text import TfidfVectorizer


DATA_DIR = 'data'
GROUP_FILE = 'groups.json.gz'
EVENT_FILE = 'events.json.gz'
VENUE_FILE = 'venues.json.gz'
TOPIC_FILE = 'category_topics.txt'
CITY_FILE = 'cities.txt'

# use a hidden directory to save intermediate results
HIDDEN_DIR = os.path.join(DATA_DIR, '.pkl')
if not os.path.exists(HIDDEN_DIR):
    os.mkdir(HIDDEN_DIR)


# -------------- helper functions ----------------------
def _load_category_topic_mapping():
    mapping = {}
    file = os.path.join(DATA_DIR, TOPIC_FILE)
    with open(file, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split()
            mapping[int(k)] = v
    return mapping


def _load_group_descriptions(group_num=None, save_file=None):
    """
    read groups.json.gz to extract meetup group descriptions and categories
    """
    category_topic_mapping = _load_category_topic_mapping()
    descriptions, topics = [], []
    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in islice(f, group_num)):
            dscr = group.get('description')
            ctg = group.get('category')
            if dscr and ctg:
                descriptions.append(dscr)
                topics.append(category_topic_mapping[ctg['id']])
    # save pickle
    pickle.dump([descriptions, topics],
                open(save_file, 'wb'))
    return descriptions, topics


def _get_group_topic_mapping(save_file):
    """
    read group.json.gz to extract id and category
    """
    category_topic_mapping = _load_category_topic_mapping()
    group_topic_mapping = {}
    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in f):
            try:
                group_topic_mapping[group['id']] \
                    = category_topic_mapping[group['category']['id']]
            except KeyError:
                pass
    # save pickle
    pickle.dump(group_topic_mapping,
                open(save_file, 'wb'))
    return group_topic_mapping


def _select_maxlen(examine, descriptions):
    similarity_items = list(set(chain(*examine)))
    dscr_len = {i: len(descriptions[i]) for i in similarity_items}
    dscr_maxlen = max(dscr_len, key=dscr_len.get)
    return similarity_items, dscr_maxlen


def get_unique(descriptions, threshold=0.5):
    try:
        tfidf = TfidfVectorizer().fit_transform(descriptions)
    except ValueError:
        print(descriptions)
        return []

    similarity = (tfidf * tfidf.T).A
    similar_pairs = np.where(np.triu(similarity, 1) > threshold)
    similar_pairs = list(zip(*similar_pairs))
    if not similar_pairs:
        return descriptions

    unique_ids = []
    examine, examined = [similar_pairs[0]], []
    for i, pair in enumerate(similar_pairs[1:]):
        if examine and (pair[0] != examine[-1][0]):
            similarity_items, dscr_maxlen = _select_maxlen(examine, descriptions)
            unique_ids.append(dscr_maxlen)
            examined.extend(similarity_items)
            examine = []

        if (pair[0] not in examined) and (pair[1] not in examined):
            examine.append(pair)

    if examine:
        similarity_items, dscr_maxlen = _select_maxlen(examine, descriptions)
        unique_ids.append(dscr_maxlen)
        examined.extend(similarity_items)
        examine = []

    unique_ids += list(set(range(len(descriptions))).difference(set(examined)))

    return [descriptions[i] for i in unique_ids]


def _load_event_descriptions(event_num=None):
    """

    """
    descriptions, topics = [], []
    group_ids, dscr_tracking = set(), set()
    current_id = None
    mapping = get_group_topic_mapping()

    file = os.path.join(DATA_DIR, EVENT_FILE)
    with gzip.open(file) as f:
        for i, event in enumerate((json.loads(x) for x in islice(f, event_num))):
            if i % 1000 == 0:
                print(i)
            dscr = event.get("description")
            if dscr:
                dscr = h.clean_text(dscr)
            grpId = event.get("group").get('id')

            if not (dscr and grpId and (grpId in mapping)):
                continue

            if current_id and dscr_tracking and (grpId != current_id):
                unique_descriptions = get_unique(list(dscr_tracking),
                                                 threshold=0.5)
                topic = [mapping[current_id]] * len(unique_descriptions)

                descriptions.extend(unique_descriptions)
                topics.extend(topic)

                group_ids.add(grpId)
                current_id = grpId
                dscr_tracking = set()

            if not current_id:
                current_id = grpId
            dscr_tracking.add(dscr)

    return descriptions, topics


def _get_topic_mapping(topics, file_path):
    counts = Counter(topics)
    ranked = sorted(counts, key=counts.get, reverse=True)
    mapping = {ctg: i for i, ctg in enumerate(ranked)}
    pickle.dump(mapping, open(file_path, 'wb'))
    return mapping


# ------------- main functions -----------------
def load_group_descriptions(group_num=None):
    """
    load group descriptions and categories
    """
    file = os.path.join(HIDDEN_DIR, 'group_descriptions')
    try:
        return pickle.load(open(file, 'rb'))
    except FileNotFoundError:
        return _load_group_descriptions(save_file=file)


def get_group_topic_mapping():
    """
    load a dictionary of {group_id: category} mapping
    """
    file = os.path.join(HIDDEN_DIR, 'group_topic_mapping')
    try:
        return pickle.load(open(file, 'rb'))
    except FileNotFoundError:
        return _get_group_topic_mapping(file)


def load_event_descriptions(event_num=None):
    """
    load event descriptions and categories
    """
    loadpath = os.path.join(DATA_DIR, "event_dscr")
    try:
        descriptions, topics = pickle.load(
            open(loadpath, "rb"))
    except FileNotFoundError:
        descriptions, topics = _load_event_descriptions(event_num)
    return descriptions, topics


def label_transform(topics):
    """
    transform topic names to ids using topic_mapping
    """
    file_path = os.path.join(HIDDEN_DIR, 'topic_mapping')
    try:
        mapping = pickle.load(open(file_path, 'rb'))
    except FileNotFoundError:
        mapping = _get_topic_mapping(topics, file_path)
    labels = [mapping[t] for t in topics]
    return labels


def prepare_dateset(descriptions, labels, topic_ids=[], sample_balance=False):
    """
    get pairs of descriptions and labels with requested ids
    input:
        topic_ids: e.g. [1:5] (top 5 topics);
        sample_balance: balance the number of groups in each category
    """
    indicies = {l: [] for l in topic_ids}
    for idx, (dscrp, label) in enumerate(zip(descriptions, labels)):
        if label in topic_ids:
            indicies[label].append(idx)

    selected = []
    if sample_balance:
        sample_N = min(map(lambda x: len(x), indicies.values()))
        for label in topic_ids:
            selected.extend(random.sample(indicies[label], sample_N))
    else:
        selected = list(chain(*list(indicies.values())))

    new_descriptions = [descriptions[i] for i in selected]
    new_labels = [labels[i] for i in selected]
    return new_descriptions, new_labels
