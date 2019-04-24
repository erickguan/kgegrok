import pprint
import sys
import logging
import json
import os
import functools
from collections import defaultdict
from typing import Iterable, Callable

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
      raise RuntimeError("Invalid key {} for rank extraction".format(key))

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

  def __init__(self, stats: Iterable[StatTool]=None, after_gathers: Iterable[StatTool]=None):
    """creates a stat gather with certain statistical tools."""
    self._gathers = []
    self._after_gathers = []
    if stats is not None:
      for s in stats: self.add_stat(s)
    if after_gathers is not None:
      for h in after_gathers: self.add_after_gather(h)

  def add_stat(self, stat: StatTool):
    self._gathers.append(stat)

  def add_after_gather(self, after_gather: Callable[[dict], None]):
    """After gathers has the access to (results, rank_results, epoch).
    It should not change the data."""
    self._after_gathers.append(after_gather)

  def __call__(self, rank_results, epoch=None):
    results = functools.reduce(lambda reduced, stat: stat(reduced, rank_results),
      self._gathers,
      {}
    )
    for h in self._after_gathers:
      h(results, rank_results, epoch)

    return results


def print_hook_after_stat_epoch():
  """The hook to print after each epoch in StatGather."""

  def hook(results, _, epoch):
    pprint.pprint(epoch)
    pprint.pprint(results)
    sys.stdout.flush()
  return hook


class ReportDrawer(object):
  """It's actually expecting Visdom drawer."""

  @staticmethod
  def default_drawer_options(config, title=None):
    if title is not None:
      title = "{}/{}".format(config.name, title)
    return dict(fillarea=True, xlabel="Epoch", width=600, height=600, title=title)

  def __init__(self, drawer, config):
    self.drawer = drawer
    # holds the our key: visdom window key
    self.plots = {}
    self.plots_opts = {}
    self.config = config

  def _is_plot_exist(self, key):
    return key in self.plots_opts

  def hook_after_stat_epoch(self):
    """returns a function to use as the hook for StatGather."""

    def fn(results, _, epoch):
      if epoch is None: return
      for key, value in results.items():
        self.append(
            key,
            X=np.array([epoch], dtype='i'),
            Y=np.array([value], dtype='f'))

    return fn

  def _lazy_create_plot(self, key, X, Y):
    """Return False if creation is not needed."""
    if key not in self.plots:
      if key in self.plots_opts:
        options = self.plots_opts[key]
      else:
        options = ReportDrawer.default_drawer_options(self.config)
      self.plots[key] = self.drawer.line(X=X, Y=Y, opts=options)
      return True
    return False

  def append(self, key, X, Y):
    """X and Y are numpy array"""
    if not self._lazy_create_plot(key, X, Y):
      # append condition
      self.drawer.line(X=X, Y=Y, win=self.plots[key], update='append')

  def create_plot_opts(self, key, options=None, use_default=True):
    # lazy creation so we can avoid an empty record in the beginning
    if use_default:
      opt = ReportDrawer.default_drawer_options(self.config)
      opt.update(options)
    else:
      opt = options
    self.plots_opts[key] = opt

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

def create_drawer(config):
  import visdom

  return ReportDrawer(visdom.Visdom(
      port=6006, log_to_filename=os.path.join(config.logging_path, config.name)), config) if config.plot_graph else None


def report_prediction_result(stat_gather,
                             results,
                             epoch=None):
  return ret_values
