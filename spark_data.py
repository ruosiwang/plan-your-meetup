import gzip
import os
import re
import ujson as json
from itertools import islice
from datetime import datetime

DATA_DIR = 'data'


def prepare_spark_data(data_type, part_num=200, max_num=2000):
    dir_path = os.path.join(DATA_DIR, data_type)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(DATA_DIR, f'{data_type}.json.gz')

    part_i = 0
    with gzip.open(file_path) as fin:
        for group in (x for x in islice(fin, max_num)):
            if not part_i % part_num:
                part_file = f'part-{part_i // part_num:05d}.json.gz'
                part_file_path = os.path.join(dir_path, part_file)

            part_i += 1
            with gzip.open(part_file_path, 'ab') as fout:
                fout.write(group)


def localpath(path):
    return 'file://' + os.path.join(os.path.abspath(os.path.curdir), path)


def _get(item, key):
    if isinstance(item, list):
        return [i.get(key) for i in item]
    elif isinstance(item, dict):
        return item.get(key)
    else:
        return None


def _int(num):
	if num is not 0:
		return int(num)
	else:
		return None


def _float(num):
    if num is not 0:
        return float(num)
    else:
        return None


def _fromtimestamp(timestamp):
    try:
        return datetime.fromtimestamp(timestamp // 1000)
    except TypeError:
        return None

def clean_text(doc):
    """
    text_cleaning
    """
    if doc:
	    # Remove HTML tags
	    words_only = re.sub('<[^<]+?>|[^a-zA-Z]', ' ', doc)
	    # (only include words with more than 3 characters)
	    words = [word for word in words_only.lower().split() if len(word) > 3]
	    return " ".join(words)

 

class Group():
    KEYS = ['id', 'city', 'state', 'timezone', 'lon', 'lat',
                  'rating', 'description', 'members', 'created',
                  'name', 'organizer', 'category', 'who', 'topics']

    def __init__(self, group_id, city, state, timezone, lon, lat,
                 rating, description, members, created, name,
                 organizer, category, who, topics):
        self.id = group_id
        self.city = city
        self.state = state
        self.timezone = timezone
        self.name = name
        self.lon = _float(lon)
        self.lat = _float(lat)
        self.rating = _float(rating)
        self.members = _int(members)
        self.created = _fromtimestamp(created)
        self.description = clean_text(description)
        self.organizer = _get(organizer, 'member_id')
        self.category = _get(category, 'shortname')
        self.topics = _get(topics, 'urlkey')
        self.who = who

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Event():
    KEYS = ['id', 'name', 'venue', 'rating', 'event_hosts',
            'yes_rsvp_count', 'maybe_rsvp_count', 'waitlist_count',
            'description', 'group', 'created', 'time', 'updated']

    def __init__(self, event_id, name, venue, rating, hosts,
                 yes_rsvp_count, maybe_rsvp_count, waitlist_count,
                 description, group, created, time, updated):
        self.id = event_id
        self.name = name
        self.group_id = _get(group, 'id')
        self.venue_id = _get(venue, 'id')
        self.rating_count = _int(_get(rating, 'count'))
        self.rating = _float(_get(rating, 'average'))
        self.yes_rsvp = _int(yes_rsvp_count)
        self.maybe_rsvp = _int(maybe_rsvp_count)
        self.waitlist = _int(waitlist_count)
        self.description = clean_text(description)
        self.hosts = _get(hosts, 'member_id')
        self.time = _fromtimestamp(time)
        self.created = _fromtimestamp(created)
        self.updated = _fromtimestamp(updated)

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Rsvp():
    KEYS = ['rsvp_id', 'response', 'member', 'event',
            'venue', 'group', 'created', 'mtime']

    def __init__(self, rsvp_id, response, member,
                 event, venue, group, created, mtime):
        self.id = rsvp_id
        self.response = response
        self.member_id = _get(member, 'member_id')
        self.event_id = _get(event, 'id')
        self.group_id = _get(group, 'id')
        self.venue_id = _get(venue, 'id')
        self.created = _fromtimestamp(created)
        self.mtime = _fromtimestamp(mtime)

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Venue():
    KEYS = ['id', 'name', 'rating', 'rating_count',
            'lon', 'lat', 'address_1', 'zip', 'city', 'state']

    def __init__(self, venue_id, name, rating, rating_count,
                 lon, lat, address_1, zip_code, city, state):
        self.id = venue_id
        self.name = name
        self.rating = _float(rating)
        self.rating_count = _int(rating_count)
        self.lon = _float(lon)
        self.lat = _float(lat)
        self.address = address_1
        self.zip = zip_code
        self.city = city
        self.state = state

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


if __name__ == '__main__':
	prepare_spark_data("groups", part_num=2000, max_num=None)
	prepare_spark_data("venues", part_num=4000, max_num=None)
	prepare_spark_data("events", part_num=2000, max_num=None)

