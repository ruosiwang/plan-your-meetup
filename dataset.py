import os
import gzip
import random
import pickle
import ujson as json
import itertools
import helper as h
import numpy as np

from collections import Counter
from itertools import islice, chain

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = 'data'
GROUP_FILE = 'groups.json.gz'
EVENT_FILE = 'events.json.gz'
VENUE_FILE = 'venues.json.gz'


def _load_group_descriptions(group_num=None, save_file=None):
    """

    """
    group_ids = set()  # make sure no replicates
    descriptions = []  # a list of group descriptions
    categories = []  # a list of group categories

    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in islice(f, group_num)):
            dscr = group.get("description")
            ctg = group.get("category")
            grpId = group.get("id")

            if all([dscr, ctg, grpId]) and (grpId not in group_ids):
                descriptions.append(dscr)
                categories.append(ctg.get("shortname"))
                group_ids.add(grpId)
    # save pickle
    pickle.dump([descriptions, categories],
                open(save_file, 'wb'))
    return descriptions, categories


def load_group_descriptions(group_num=None):
    file = os.path.join(DATA_DIR, 'group_dscr')
    try:
        return pickle.load(open(file, 'rb'))
    except FileNotFoundError:
        return _load_group_descriptions(save_file=file)


def _get_group_category_mapping():
    group_category_mapping = {}
    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in f):
            ctg = group.get("category")
            if not ctg:
                grpId = group.get("id")
                group_category_mapping[grpId] = ctg
    return group_category_mapping


def get_group_category_mapping():
    file = os.path.join(DATA_DIR, 'group_category_mapping')
    try:
        return pickle.load(open(file, 'rb'))
    except FileNotFoundError:
        return _get_group_category_mapping()


def _select_maxlen(examine, descriptions):
    similarity_items = list(set(itertools.chain(*examine)))
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


def load_event_descriptions(event_num=None):
    loadpath = os.path.join(DATA_DIR, "event_dscr")
    try:
        descriptions, categories = pickle.load(
            open(loadpath, "rb"))
    except FileNotFoundError:
        descriptions, categories = _load_event_descriptions(event_num)
    return descriptions, categories


def _load_event_descriptions(event_num=None):
    """

    """
    descriptions, categories = [], []
    group_ids, dscr_tracking = set(), set()
    current_id = None
    mapping = get_group_category_mapping()

    file = os.path.join(DATA_DIR, EVENT_FILE)
    with gzip.open(file) as f:
        for i, event in enumerate((json.loads(x) for x in islice(f, event_num))):
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
                category = [mapping[current_id]] * len(unique_descriptions)

                descriptions.extend(unique_descriptions)
                categories.extend(category)

                group_ids.add(grpId)
                current_id = grpId
                dscr_tracking = set()

            if not current_id:
                current_id = grpId
            dscr_tracking.add(dscr)

    return descriptions, categories

    # # save pickle
    # return descriptions, categories


def load_group(field_list):
    """

    """
    group_ids = set()  # make sure no replicates
    info = []

    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in f):
            grpId = group.get('id')
            this = [group.get(k) for k in field_list]

            if all(this) and (grpId not in group_ids):
                info.append(this)
                group_ids.add(grpId)

    # save pickle
    return info


def label_transform(categories):
    counts = Counter(categories)
    ranks = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    mapping = {t[0]: i for i, t in enumerate(ranks)}
    labels = list(map(lambda cat: mapping[cat], categories))
    return labels, mapping


def prepare_dateset(descriptions, labels, category_num=10, sample_balance=True):

    indicies = {l: [] for l in range(0, category_num)}
    for idx, (dscrp, label) in enumerate(zip(descriptions, labels)):
        if label < category_num:
            indicies[label].append(idx)

    selected = []
    if sample_balance:
        sample_N = labels.count(category_num)
        for label in range(0, category_num):
            selected.extend(random.sample(indicies[label], sample_N))
    else:
        selected = list(chain(*list(indicies.values())))

    new_descriptions = [descriptions[i] for i in selected]
    new_labels = [labels[i] for i in selected]
    return new_descriptions, new_labels
