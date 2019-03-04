import pprint
import sys
import logging
import json
import functools
from collections import defaultdict
from typing import Iterable

import numpy as np

import kgekit
from kgegrok.data import constants
from kgegrok.data import statstools
from kgegrok.stats.constants import *


class StatTool(object):
  @staticmethod
  def extract_ranks(ranks, key, filtered: bool):
    """Extarct ranks from rank results."""
    hr, fhr, tr, ftr, rr, frr = ranks
    if key == constants.HEAD_KEY:
      return fhr if filtered else hr
    elif key == constants.TAIL_KEY:
      return ftr if filtered else tr
    elif key == constants.RELATION_KEY:
      return frr if filtered else rr
    else:
      raise RuntimeError("Invalid key for rank extraction")

  @staticmethod
  def gen_key(*argv):
    """generate the statistical tool's result."""
    return statstools.dict_key_gen(*tuple(map(str, argv)))

  def __call__(self, results, ranks):
    raise NotImplementedError()


class ElementHitStatTool(StatTool):
  """Calculate hits on an element in triples."""
  def __init__(self, entity_key, hit_level, filtered=False):
    self._entity_key = entity_key
    self._hit_level = hit_level
    self._filtered = filtered
    self._prefix_key = FILTERED_HITS_FEATURE_PREFIX if self._filtered else HITS_FEATURE_PREFIX
    self._result_key = StatTool.gen_key(self._entity_key, self._prefix_key, self._hit_level)

  def __call__(self, results, ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(ranks, self._entity_key, filtered=self._filtered)

    value = statstools.calc_hits(self._hit_level, ranks, len(ranks))

    results[self._result_key] = value
    return results

class CombinedEntityHitStatTool(ElementHitStatTool):
  """Calculate hits on entities in a triple."""
  def __init__(self, entity_key, hit_level, filtered=False):
    super().__init__(entity_key, hit_level, filtered=filtered)

  def __call__(self, results, result_ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(result_ranks, constants.HEAD_KEY, filtered=self._filtered)
    head_result = statstools.calc_hits(self._hit_level, ranks, len(ranks))
    ranks = StatTool.extract_ranks(result_ranks, constants.TAIL_KEY, filtered=self._filtered)
    tail_result = statstools.calc_hits(self._hit_level, ranks, len(ranks))
    value = (head_result + tail_result) / 2.0

    results[self._result_key] = value
    return results

class ElementMeanRankStatTool(StatTool):
  """Calculate mean rank on an element in triples."""

  def __init__(self, entity_key, filtered=False):
    self._entity_key = entity_key
    self._filtered = filtered
    self._prefix_key = MEAN_FILTERED_RANK_FEATURE_KEY if self._filtered else MEAN_RANK_FEATURE_KEY
    self._result_key = StatTool.gen_key(self._entity_key, self._prefix_key)


  def __call__(self, results, ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(ranks, self._entity_key, filtered=self._filtered)

    value = statstools.calc_rank(ranks, len(ranks))

    results[self._result_key] = value
    return results

class CombinedEntityMeanRankStatTool(ElementMeanRankStatTool):
  """Calculate mean rank on two entities in triples."""

  def __init__(self, entity_key, filtered=False):
    super().__init__(entity_key, filtered=filtered)

  def __call__(self, results, result_ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(result_ranks, constants.HEAD_KEY, filtered=self._filtered)
    head_result = statstools.calc_rank(ranks, len(ranks))
    ranks = StatTool.extract_ranks(result_ranks, constants.TAIL_KEY, filtered=self._filtered)
    tail_result = statstools.calc_rank(ranks, len(ranks))
    value = (head_result + tail_result) / 2.0

    results[self._result_key] = value
    return results


class ElementMeanReciprocalRankStatTool(StatTool):
  """Calculate mean reciprocal rank on an element in triples."""

  def __init__(self, entity_key, filtered=False):
    self._entity_key = entity_key
    self._filtered = filtered
    self._prefix_key = MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY if self._filtered else MEAN_RECIPROCAL_RANK_FEATURE_KEY
    self._result_key = StatTool.gen_key(self._entity_key, self._prefix_key)

  def __call__(self, results, ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(ranks, self._entity_key, filtered=self._filtered)

    value = statstools.calc_reciprocal_rank(ranks, len(ranks))

    results[self._result_key] = value
    return results

class CombinedEntityMeanReciprocalRankStatTool(ElementMeanReciprocalRankStatTool):
  """Calculate mean reciprocal rank on two entities in triples."""

  def __init__(self, entity_key, filtered=False):
    super().__init__(entity_key, filtered=filtered)

  def __call__(self, results, result_ranks):
    """Extract ranks from result triples. Applies with the calculation function.
    And then stored in a dict with key defined."""
    ranks = StatTool.extract_ranks(result_ranks, constants.HEAD_KEY, filtered=self._filtered)
    head_result = statstools.calc_reciprocal_rank(ranks, len(ranks))
    ranks = StatTool.extract_ranks(result_ranks, constants.TAIL_KEY, filtered=self._filtered)
    tail_result = statstools.calc_reciprocal_rank(ranks, len(ranks))
    value = (head_result + tail_result) / 2.0

    results[self._result_key] = value
    return results

class StatGather(object):
  """Collects stats from ranking data."""

  def __init__(self, stats: Iterable[StatTool] = None):
    """creates a stat gather with certain statistical tools."""
    self._gathers = []
    if stats is not None:
      for s in stats: self.add_stat(s)

  def add_stat(self, stat: StatTool):
    self._gathers.append(stat)

  def __call__(self, rank_results):
    return functools.reduce(lambda reduced, stat: stat(reduced, rank_results),
      self._gathers,
      {}
    )

def get_evaluation_statistics(rank_list, filtered_rank_list, features):
  num_ranks = len(rank_list)
  assert isinstance(rank_list, list) and isinstance(
      filtered_rank_list, list) and num_ranks == len(filtered_rank_list)

  gathering = StatGather()
  if LinkPredictionStatistics.MEAN_RECIPROCAL_RANK & features:
    gathering.add_reciprocal_rank(MEAN_RECIPROCAL_RANK_FEATURE_KEY, rank_list,
                                  num_ranks)
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
  return dict(fillarea=True, xlabel="Epoch", width=600, height=600, title=title)


def _report_prediction_element(element, epoch):
  pprint.pprint(epoch)
  pprint.pprint(element)
  sys.stdout.flush()


def _append_drawer(drawer, epoch, result, prefix_key=None):
  assert epoch is not None
  for key, value in result.items():
    drawer_key = statstools.dict_key_gen(prefix_key,
                                         key) if prefix_key is not None else key
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
    combined = {k: (h + tail_result[k]) / 2.0 for k, h in head_result.items()}
    ret_values[constants.ENTITY_KEY] = combined
    _report_prediction_element(combined, epoch)
    if drawer is not None:
      _append_drawer(drawer, epoch, combined)

  if config.report_dimension & StatisticsDimension.RELATION:
    relation_result = get_evaluation_statistics(rr, frr, config.report_features)
    ret_values[constants.RELATION_KEY] = relation_result
    _report_prediction_element(relation_result, epoch)
    if drawer is not None:
      _append_drawer(drawer, epoch, relation_result, constants.RELATION_KEY)

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

  def log_to_filename(self, log_path):
    self.drawer.log_to_filename(log_to_filename)

  def dump_raw_data(self):
    """Dumps all raw data."""

    raw_data = []
    for _, win in self.plots.items():
      raw_data.append(self._dump_win_data(win))
    return raw_data


def prepare_plot_validation_result(drawer, config):
  if config.report_dimension & StatisticsDimension.COMBINED_ENTITY:
    for key, condition in DRAWING_KEY_AND_CONDITION[
        StatisticsDimension.COMBINED_ENTITY]:
      if condition & config.report_features:
        drawer.create_plot(key, gen_drawer_option(config, key))
  elif config.report_dimension & StatisticsDimension.SEPERATE_ENTITY:
    for key, condition in DRAWING_KEY_AND_CONDITION[
        StatisticsDimension.SEPERATE_ENTITY]:
      if condition & config.report_features:
        drawer.create_plot(key, gen_drawer_option(config, key))
  if config.report_dimension & StatisticsDimension.RELATION:
    for key, condition in DRAWING_KEY_AND_CONDITION[
        StatisticsDimension.RELATION]:
      if condition & config.report_features:
        drawer.create_plot(key, gen_drawer_option(config, key))

  return drawer


def create_drawer(config):
  import visdom

  return ReportDrawer(visdom.Visdom(
      port=6006), config) if config.plot_graph else None
