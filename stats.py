import pprint
import data
import kgekit
import numpy as np
import logging
from collections import defaultdict
import json
from enum import Flag, auto


MEAN_RECIPROCAL_RANK_FEATURE_KEY = 'mean_reciprocal_rank'
MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY = 'mean_filtered_reciprocal_rank'
MEAN_RANK_FEATURE_KEY = 'mean_rank'
MEAN_FILTERED_RANK_FEATURE_KEY = 'mean_filtered_rank'
HITS_1_FEATURE_KEY = 'hits_1'
HITS_1_FILTERED_FEATURE_KEY = "hits_1_filtered"
HITS_3_FEATURE_KEY = 'hits_3'
HITS_3_FILTERED_FEATURE_KEY = 'hits_3_filtered'
HITS_5_FEATURE_KEY = 'hits_5'
HITS_5_FILTERED_FEATURE_KEY = "hits_5_filtered"
HITS_10_FEATURE_KEY = 'hits_10'
HITS_10_FILTERED_FEATURE_KEY = "hits_10_filtered"

class LinkPredictionStatistics(Flag):
    MEAN_RECIPROCAL_RANK = auto()
    MEAN_FILTERED_RECIPROCAL_RANK = auto()
    MEAN_RANK = auto()
    MEAN_FILTERED_RANK = auto()
    MEAN_RANKS = MEAN_RECIPROCAL_RANK | MEAN_RANK
    MEAN_FILTERED_RANKS = MEAN_FILTERED_RECIPROCAL_RANK | MEAN_FILTERED_RANK
    ALL_RANKS = MEAN_RANKS | MEAN_FILTERED_RANKS
    HITS_1 = auto()
    HITS_1_FILTERED = auto()
    HITS_3 = auto()
    HITS_3_FILTERED = auto()
    HITS_5 = auto()
    HITS_5_FILTERED = auto()
    HITS_10 = auto()
    HITS_10_FILTERED = auto()
    MINIMAL_HITS = HITS_1 | HITS_10
    MINIMAL_HITS_FILTERED = HITS_1_FILTERED | HITS_10_FILTERED
    HITS = HITS_1 | HITS_3 | HITS_5 | HITS_10
    HITS_FILTERED = HITS_1_FILTERED | HITS_3_FILTERED | HITS_5_FILTERED | HITS_10_FILTERED
    ALL_HITS = HITS | HITS_FILTERED
    UNFILTERED_DEFAULT = MINIMAL_HITS | MEAN_RECIPROCAL_RANK
    DEFAULT = HITS_FILTERED | MEAN_FILTERED_RECIPROCAL_RANK
    ALL = ALL_RANKS | ALL_HITS

class StatisticsDimension(Flag):
    SEPERATE_ENTITY = auto()
    COMBINED_ENTITY = auto()
    RELATION = auto()
    DEFAULT = COMBINED_ENTITY | RELATION
    ALL = SEPERATE_ENTITY | RELATION

class _StatisticsGathering(object):
    def __init__(self):
        self.result = {}

    def add_rank(self, key, ranks, num_ranks):
        self.result[key] = data.calc_rank(ranks, num_ranks)

    def add_reciprocal_rank(self, key, ranks, num_ranks):
        self.result[key] = data.calc_reciprocal_rank(ranks, num_ranks)

    def add_hit(self, key, ranks, target, num_ranks):
        self.result[key] = data.calc_hits(target, ranks, num_ranks)

    def get_result(self):
        return self.result


def get_evaluation_statistics(rank_list, filtered_rank_list, features):
    num_ranks = len(rank_list)
    assert isinstance(rank_list, list) and isinstance(filtered_rank_list, list) and num_ranks == len(filtered_rank_list)

    gathering = _StatisticsGathering()
    if LinkPredictionStatistics.MEAN_RECIPROCAL_RANK & features:
        gathering.add_reciprocal_rank(MEAN_RECIPROCAL_RANK_FEATURE_KEY, rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK & features:
        gathering.add_reciprocal_rank(MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY, filtered_rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_RANK & features:
        gathering.add_rank(MEAN_RANK_FEATURE_KEY, rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_FILTERED_RANK & features:
        gathering.add_rank(MEAN_FILTERED_RANK_FEATURE_KEY, filtered_rank_list, num_ranks)
    if LinkPredictionStatistics.HITS_1 & features:
        gathering.add_hit(HITS_1_FEATURE_KEY, rank_list, 1, num_ranks)
    if LinkPredictionStatistics.HITS_3 & features:
        gathering.add_hit(HITS_3_FEATURE_KEY, rank_list, 3, num_ranks)
    if LinkPredictionStatistics.HITS_5 & features:
        gathering.add_hit(HITS_5_FEATURE_KEY, rank_list, 5, num_ranks)
    if LinkPredictionStatistics.HITS_10 & features:
        gathering.add_hit(HITS_10_FEATURE_KEY, rank_list, 10, num_ranks)
    if LinkPredictionStatistics.HITS_1_FILTERED & features:
        gathering.add_hit(HITS_1_FILTERED_FEATURE_KEY, filtered_rank_list, 1, num_ranks)
    if LinkPredictionStatistics.HITS_3_FILTERED & features:
        gathering.add_hit(HITS_3_FILTERED_FEATURE_KEY, filtered_rank_list, 3, num_ranks)
    if LinkPredictionStatistics.HITS_5_FILTERED & features:
        gathering.add_hit(HITS_5_FILTERED_FEATURE_KEY, filtered_rank_list, 5, num_ranks)
    if LinkPredictionStatistics.HITS_10_FILTERED & features:
        gathering.add_hit(HITS_10_FILTERED_FEATURE_KEY, filtered_rank_list, 10, num_ranks)
    return gathering.get_result()


def gen_drawer_option(config, title=None):
    if title is not None:
        title = "{}/{}".format(config.name, title)
    return dict(fillarea=True, xlabel="Epoch", width=600, height=600, title=title)

def _report_prediction_element(element, epoch):
    pprint.pprint(epoch)
    pprint.pprint(element)

def _append_drawer(drawer, epoch, result, prefix_key=None):
    assert epoch is not None
    for key, value in result.items():
        drawer_key = data.dict_key_gen(prefix_key, key) if prefix_key is not None else key
        drawer.append(drawer_key, X=np.array([epoch], dtype='f'), Y=np.array([value], dtype='f'))

def report_prediction_result(config, result, printing=True, epoch=None, drawer=None):
    heads, tails, relations = result
    ret_values = {}

    if config.report_dimension & StatisticsDimension.SEPERATE_ENTITY:
        head_result = get_evaluation_statistics(*heads, config.report_features)
        tail_result = get_evaluation_statistics(*tails, config.report_features)
        ret_values[data.HEAD_KEY] = head_result
        ret_values[data.TAIL_KEY] = tail_result
        _report_prediction_element(head_result, epoch)
        _report_prediction_element(tail_result, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, head_result, data.HEAD_KEY)
            _append_drawer(drawer, epoch, tail_result, data.TAIL_KEY)

    elif config.report_dimension & StatisticsDimension.COMBINED_ENTITY:
        head_result = get_evaluation_statistics(*heads, config.report_features)
        tail_result = get_evaluation_statistics(*tails, config.report_features)
        combined = {k: (h + tail_result[k]) / 2.0 for k, h in head_result.items()}
        ret_values[data.ENTITY_KEY] = combined
        _report_prediction_element(combined, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, combined)

    if config.report_dimension & StatisticsDimension.RELATION:
        relation_result = get_evaluation_statistics(*relations, config.report_features)
        ret_values[data.RELATION_KEY] = relation_result
        _report_prediction_element(relation_result, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, relation_result, data.RELATION_KEY)

    return ret_values

class ReportDrawer(object):
    """It's actually expecting Visdom drawer."""

    def __init__(self, drawer, config):
        self.drawer = drawer
        # holds the our key: visdom window key
        self.plots = {}
        self.plots_opts = {}
        self.config = config

    def _is_plot_exist(self, key):
        return key in self.plots_opts

    def _lazy_create_plot(self, key, X, Y):
        """Return False if creation is not needed."""
        if key not in self.plots:
            self.plots[key] = self.drawer.line(X=X, Y=Y, opts=self.plots_opts[key])
            return True
        return False

    def append(self, key, X, Y):
        """X and Y are numpy array"""
        if not self._is_plot_exist:
            raise RuntimeError("{} doesn't exists in drawer.".format(key))
        if not self._lazy_create_plot(key, X, Y):
            # append condition
            self.drawer.line(X=X, Y=Y, win=self.plots[key], update='append')

    def create_plot(self, key, options=None):
        # lazy creation so we can avoid an empty record in the beginning
        self.plots_opts[key] = options

    def _dump_win_data(self, win):
        content = self.drawer.get_window_data(win)
        if content is None or len(content) == 0:
            content = "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(content)
        return content

    def dump_raw_data(self):
        """Dumps all raw data."""

        raw_data = []
        for _, win in self.plots.items():
            raw_data.append(_dump_win_data(win))
        return raw_data


DRAWING_KEY_AND_CONDITION = {
StatisticsDimension.COMBINED_ENTITY: [
    (MEAN_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_RANK),
    (MEAN_FILTERED_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (MEAN_RECIPROCAL_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (HITS_1_FEATURE_KEY, LinkPredictionStatistics.HITS_1),
    (HITS_3_FEATURE_KEY, LinkPredictionStatistics.HITS_3),
    (HITS_5_FEATURE_KEY, LinkPredictionStatistics.HITS_5),
    (HITS_10_FEATURE_KEY, LinkPredictionStatistics.HITS_10),
    (HITS_1_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_1_FILTERED),
    (HITS_3_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_3_FILTERED),
    (HITS_5_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_5_FILTERED),
    (HITS_10_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_10_FILTERED),
],
StatisticsDimension.SEPERATE_ENTITY: [
    (data.dict_key_gen(data.HEAD_KEY, MEAN_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.HEAD_KEY, MEAN_FILTERED_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.HEAD_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.HEAD_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.HEAD_KEY, HITS_1_FEATURE_KEY), LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.HEAD_KEY, HITS_3_FEATURE_KEY), LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.HEAD_KEY, HITS_5_FEATURE_KEY), LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.HEAD_KEY, HITS_10_FEATURE_KEY), LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.HEAD_KEY, HITS_1_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, HITS_3_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, HITS_5_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, HITS_10_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_10_FILTERED),

    (data.dict_key_gen(data.TAIL_KEY, MEAN_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.TAIL_KEY, MEAN_FILTERED_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.TAIL_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.TAIL_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.TAIL_KEY, HITS_1_FEATURE_KEY), LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.TAIL_KEY, HITS_3_FEATURE_KEY), LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.TAIL_KEY, HITS_5_FEATURE_KEY), LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.TAIL_KEY, HITS_10_FEATURE_KEY), LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.TAIL_KEY, HITS_1_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, HITS_3_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, HITS_5_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, HITS_10_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_10_FILTERED),
],
StatisticsDimension.RELATION: [
    (data.dict_key_gen(data.RELATION_KEY, MEAN_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.RELATION_KEY, MEAN_FILTERED_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.RELATION_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.RELATION_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.RELATION_KEY, HITS_1_FEATURE_KEY), LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.RELATION_KEY, HITS_3_FEATURE_KEY), LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.RELATION_KEY, HITS_5_FEATURE_KEY), LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.RELATION_KEY, HITS_10_FEATURE_KEY), LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.RELATION_KEY, HITS_1_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, HITS_3_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, HITS_5_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, HITS_10_FILTERED_FEATURE_KEY), LinkPredictionStatistics.HITS_10_FILTERED),
]
}

def prepare_plot_validation_result(drawer, config):
    if config.report_dimension & StatisticsDimension.COMBINED_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.COMBINED_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    elif config.report_dimension & StatisticsDimension.SEPERATE_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.SEPERATE_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    if config.report_dimension & StatisticsDimension.RELATION:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.RELATION]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))

    return drawer
