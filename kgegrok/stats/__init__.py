import pprint
import sys
import logging
import json
from collections import defaultdict

import numpy as np

import kgekit
from kgegrok.data import constants
from kgegrok.data import statstools
from kgegrok.stats.constants import *


class _StatisticsGathering(object):

    def __init__(self):
        self.result = {}

    def add_rank(self, key, ranks, num_ranks):
        self.result[key] = statstools.calc_rank(ranks, num_ranks)

    def add_reciprocal_rank(self, key, ranks, num_ranks):
        self.result[key] = statstools.calc_reciprocal_rank(ranks, num_ranks)

    def add_hit(self, key, ranks, target, num_ranks):
        self.result[key] = statstools.calc_hits(target, ranks, num_ranks)

    def get_result(self):
        return self.result


def get_evaluation_statistics(rank_list, filtered_rank_list, features):
    num_ranks = len(rank_list)
    assert isinstance(rank_list, list) and isinstance(
        filtered_rank_list, list) and num_ranks == len(filtered_rank_list)

    gathering = _StatisticsGathering()
    if LinkPredictionStatistics.MEAN_RECIPROCAL_RANK & features:
        gathering.add_reciprocal_rank(MEAN_RECIPROCAL_RANK_FEATURE_KEY,
                                      rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK & features:
        gathering.add_reciprocal_rank(MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY,
                                      filtered_rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_RANK & features:
        gathering.add_rank(MEAN_RANK_FEATURE_KEY, rank_list, num_ranks)
    if LinkPredictionStatistics.MEAN_FILTERED_RANK & features:
        gathering.add_rank(MEAN_FILTERED_RANK_FEATURE_KEY, filtered_rank_list,
                           num_ranks)
    if LinkPredictionStatistics.HITS_1 & features:
        gathering.add_hit(HITS_1_FEATURE_KEY, rank_list, 1, num_ranks)
    if LinkPredictionStatistics.HITS_3 & features:
        gathering.add_hit(HITS_3_FEATURE_KEY, rank_list, 3, num_ranks)
    if LinkPredictionStatistics.HITS_5 & features:
        gathering.add_hit(HITS_5_FEATURE_KEY, rank_list, 5, num_ranks)
    if LinkPredictionStatistics.HITS_10 & features:
        gathering.add_hit(HITS_10_FEATURE_KEY, rank_list, 10, num_ranks)
    if LinkPredictionStatistics.HITS_1_FILTERED & features:
        gathering.add_hit(HITS_1_FILTERED_FEATURE_KEY, filtered_rank_list, 1,
                          num_ranks)
    if LinkPredictionStatistics.HITS_3_FILTERED & features:
        gathering.add_hit(HITS_3_FILTERED_FEATURE_KEY, filtered_rank_list, 3,
                          num_ranks)
    if LinkPredictionStatistics.HITS_5_FILTERED & features:
        gathering.add_hit(HITS_5_FILTERED_FEATURE_KEY, filtered_rank_list, 5,
                          num_ranks)
    if LinkPredictionStatistics.HITS_10_FILTERED & features:
        gathering.add_hit(HITS_10_FILTERED_FEATURE_KEY, filtered_rank_list, 10,
                          num_ranks)
    return gathering.get_result()

def gen_drawer_option(config, title=None):
    if title is not None:
        title = "{}/{}".format(config.name, title)
    return dict(
        fillarea=True, xlabel="Epoch", width=600, height=600, title=title)


def _report_prediction_element(element, epoch):
    pprint.pprint(epoch)
    pprint.pprint(element)
    sys.stdout.flush()


def _append_drawer(drawer, epoch, result, prefix_key=None):
    assert epoch is not None
    for key, value in result.items():
        drawer_key = statstools.dict_key_gen(
            prefix_key, key) if prefix_key is not None else key
        drawer.append(
            drawer_key,
            X=np.array([epoch], dtype='f'),
            Y=np.array([value], dtype='f'))


def report_prediction_result(config,
                             result,
                             printing=True,
                             epoch=None,
                             drawer=None):
    hr, fhr, tr, ftr, rr, frr = result
    ret_values = {}

    if config.report_dimension & StatisticsDimension.SEPERATE_ENTITY:
        head_result = get_evaluation_statistics(hr, fhr, config.report_features)
        tail_result = get_evaluation_statistics(tr, ftr, config.report_features)
        ret_values[constants.HEAD_KEY] = head_result
        ret_values[constants.TAIL_KEY] = tail_result
        _report_prediction_element(head_result, epoch)
        _report_prediction_element(tail_result, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, head_result, constants.HEAD_KEY)
            _append_drawer(drawer, epoch, tail_result, constants.TAIL_KEY)

    elif config.report_dimension & StatisticsDimension.COMBINED_ENTITY:
        head_result = get_evaluation_statistics(hr, fhr, config.report_features)
        tail_result = get_evaluation_statistics(tr, ftr, config.report_features)
        combined = {
            k: (h + tail_result[k]) / 2.0 for k, h in head_result.items()
        }
        ret_values[constants.ENTITY_KEY] = combined
        _report_prediction_element(combined, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, combined)

    if config.report_dimension & StatisticsDimension.RELATION:
        relation_result = get_evaluation_statistics(rr, frr,
                                                    config.report_features)
        ret_values[constants.RELATION_KEY] = relation_result
        _report_prediction_element(relation_result, epoch)
        if drawer is not None:
            _append_drawer(drawer, epoch, relation_result,
                           constants.RELATION_KEY)

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
            self.plots[key] = self.drawer.line(
                X=X, Y=Y, opts=self.plots_opts[key])
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
            raw_data.append(self._dump_win_data(win))
        return raw_data


def prepare_plot_validation_result(drawer, config):
    if config.report_dimension & StatisticsDimension.COMBINED_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.
                                                        COMBINED_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    elif config.report_dimension & StatisticsDimension.SEPERATE_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.
                                                        SEPERATE_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    if config.report_dimension & StatisticsDimension.RELATION:
        for key, condition in DRAWING_KEY_AND_CONDITION[StatisticsDimension.
                                                        RELATION]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))

    return drawer

def create_drawer(config):
    import visdom

    return ReportDrawer(visdom.Visdom(
        port=6006), config) if config.plot_graph else None
