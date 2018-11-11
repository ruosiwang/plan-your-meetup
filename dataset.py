import os
import gzip
import random
import ujson as json
import numpy as np
import itertools
from collections import Counter


DATA_DIR = 'data'
GROUP_FILE = 'groups.json.gz'
EVENT_FILE = 'events.json.gz'
VENUE_FILE = 'venues.json.gz'


def load_group_descriptions():
    """

    """
    group_ids = set()  # make sure no replicates
    descriptions = []  # a list of group descriptions
    categories = []  # a list of group categories

    file = os.path.join(DATA_DIR, GROUP_FILE)
    with gzip.open(file) as f:
        for group in (json.loads(x) for x in f):
            dscr = group.get("description")
            ctg = group.get("category")
            grpId = group.get("id")

            if all([dscr, ctg, grpId]) and (grpId not in group_ids):
                descriptions.append(dscr)
                categories.append(ctg.get("shortname"))
                group_ids.add(grpId)
    # save pickle
    return descriptions, categories


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


def prepare_dateset(descriptions, labels, category_N=10, sample_balance=True):

    indicies = {l: [] for l in range(0, category_N)}
    for idx, (dscrp, label) in enumerate(zip(descriptions, labels)):
        if label < category_N:
            indicies[label].append(idx)

    selected = []
    if sample_balance:
        sample_N = labels.count(category_N)
        for label in range(0, category_N):
            selected.extend(random.sample(indicies[label], sample_N))
    else:
        selected = list(itertools.chain(*list(indicies.values())))

    new_descriptions = [descriptions[i] for i in selected]
    new_labels = [labels[i] for i in selected]
    return new_descriptions, new_labels
