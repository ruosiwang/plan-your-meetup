import numpy as np
import helper as h

import os
import pickle
import pandas as pd

from wordcloud import WordCloud
import sklearn_models as s

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.core.properties import value
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge


def explore_venues():

    color_palette = ['#ea4335', '#fbbc05', '#34a853', '#673ab7']
    categories = ['socializing', 'career-business', 'tech', 'music']

    city_lat, city_lon = (42.361145, -71.057083)

    plot_data = pd.read_pickle('boston_venue_category')
    plot_data['coords_x'] = plot_data['lon'].apply(lambda x: h.merc_x(x))
    plot_data['coords_y'] = plot_data['lat'].apply(lambda x: h.merc_y(x))

    X_range = (h.merc_x(city_lon) - 9000, h.merc_x(city_lon) + 3000)
    Y_range = (h.merc_y(city_lat) - 5000, h.merc_y(city_lat) + 5000)

    p = figure(x_range=X_range, y_range=Y_range,
               x_axis_type="mercator", y_axis_type="mercator")

    # load map
    p.add_tile(CARTODBPOSITRON)

    # add venues
    for ctg, cp in zip(categories, color_palette):
        p.circle(x=plot_data.loc[plot_data.category == ctg, 'coords_x'],
                 y=plot_data.loc[plot_data.category == ctg, 'coords_y'],
                 line_alpha=0.05,
                 fill_color=cp,
                 size=np.log(plot_data.loc[plot_data.category == ctg, 'sum(host)']) * 2.5,
                 legend=ctg,
                 fill_alpha=0.3)
    output_notebook()
    show(p)


def classification_report():
    keras_scores = pd.read_csv('models/keras_f1_scores.csv', index_col=0)
    sklearn_scores = pd.read_csv('models/sklearn_f1_scores.csv', index_col=0)
    scores = keras_scores.join(sklearn_scores)

    # prepare bokeh data
    data = {m: list(scores[m]) for m in list(scores.columns)}
    data['category'] = [f'{n} categories' for n in list(scores.index)]
    source = ColumnDataSource(data=data)
    colors = ['#ED6A5A', '#F4F1BB', '#9BC1BC', '#5CA4A9',
              '#E6EBE0', '#dcf5ff', '#e6c86e', '#508cd7']

    models = scores.T.sort_values([20], ascending=False).index[:5]
    dis = [-0.3, -0.15, 0, 0.15, 0.3]
    # dis = [-0.35, -0.25, -0.15, -0.05, .05, 0.15, 0.25, 0.35]

    p = figure(x_range=data['category'], y_range=(0.6, 1), plot_height=250, title="Category Classification Accuracy",
               toolbar_location=None, tools="")

    for i in range(len(models)):
        p.vbar(x=dodge('category', dis[i], range=p.x_range), top=models[i], width=0.12, source=source,
               color=colors[i], legend=value(models[i]))

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    output_notebook()
    show(p)


category_mapping = {'career-business': 0,
                    'tech': 1,
                    'health-wellbeing': 2,
                    'socializing': 3,
                    'outdoors-adventure': 5,
                    'sports-recreation': 6,
                    'parents-family': 7,
                    'food-drink': 9}


def _get_words(top_words, category_i):
    STOP_WORDS = ('meetup', 'join', 'group', 'members', 'events', 'people')
    words = {w: i + 10 for i, w in enumerate(top_words[category_i])}
    for stop in STOP_WORDS:
        words.pop(stop, None)
    return words


def plot_key_features(top_feature_num=30):

    files = os.listdir(s.MODEL_DIR)
    model_file = [f for f in files if f.startswith(f"MNB_10_group")][0]
    model_path = os.path.join(s.MODEL_DIR, model_file)
    model = pickle.load(open(model_path, 'rb'))

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    top_words = s.most_important_features(model, top_feature_num)

    sbp_info = list(zip([0] * 4 + [1] * 4, [0, 1, 2, 3] * 2))
    for sbpI, (ctg, label) in enumerate(category_mapping.items()):
        word_cloud = WordCloud(width=480, height=480, margin=0,
                               background_color="white").fit_words(_get_words(top_words, label))
        i, j = sbp_info[sbpI]
        axs[i][j].imshow(word_cloud, interpolation='bilinear')
        axs[i][j].axis("off")
        axs[i][j].set_title(ctg, size=20)

    plt.show()
