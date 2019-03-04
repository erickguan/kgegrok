import unittest
import pytest
from kgegrok import stats
from kgegrok.stats.constants import *
from kgegrok.data import constants, statstools

# def test_rank_stats():
#   res = stats.get_evaluation_statistics([1, 6, 11, 9, 12], [1, 5, 10, 9, 12],
#                                         LinkPredictionStatistics.ALL)
#   assert ((1.0 + 1 / 6.0 + 1 / 11.0 + 1 / 9.0 + 1 / 12.0) / 5.0 ==
#     pytest.approx(res[MEAN_RECIPROCAL_RANK_FEATURE_KEY]))
#   assert ((1.0 + 1 / 5.0 + 1 / 10.0 + 1 / 9.0 + 1 / 12.0) / 5.0 ==
#       pytest.approx(res[MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY]))
#   assert ((1 + 6 + 11 + 9 + 12) / 5.0 ==
#     pytest.approx(res[MEAN_RANK_FEATURE_KEY]))
#   assert ((1 + 5 + 10 + 9 + 12) / 5.0 ==
#     pytest.approx(res[MEAN_FILTERED_RANK_FEATURE_KEY]))
#   assert 1 / 5.0 == pytest.approx(res[HITS_1_FEATURE_KEY])
#   assert 1 / 5.0 == pytest.approx(res[HITS_3_FEATURE_KEY])
#   assert 1 / 5.0 == pytest.approx(res[HITS_5_FEATURE_KEY])
#   assert 3 / 5.0 == pytest.approx(res[HITS_10_FEATURE_KEY])
#   assert 1 / 5.0 == pytest.approx(res[HITS_1_FILTERED_FEATURE_KEY])
#   assert 1 / 5.0 == pytest.approx(res[HITS_3_FILTERED_FEATURE_KEY])
#   assert 2 / 5.0 == pytest.approx(res[HITS_5_FILTERED_FEATURE_KEY])
#   assert 4 / 5.0 == pytest.approx(res[HITS_10_FILTERED_FEATURE_KEY])

@pytest.fixture(scope="module")
def rank_results():
  """head ranks, filtered head ranks, tail ranks, filtered tail ranks, relation ranks, filtered relation ranks"""
  return ([9,3,7,11,10],
         [9,2,6,8,9],
         [7,6,8,5,4],
         [6,6,3,5,1],
         [4,9,11],
         [3,9,5])

def test_head_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.HEAD_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert 1 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_head_filtered_hit_stat_gather(rank_results):
  hit_level = 8
  gather = stats.StatGather([stats.ElementHitStatTool(constants.HEAD_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_tail_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.TAIL_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert 2 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_tail_filtered_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.TAIL_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_rel_hit_stat_gather(rank_results):
  hit_level = 9
  gather = stats.StatGather([stats.ElementHitStatTool(constants.RELATION_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_rel_filtered_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.RELATION_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_combined_hit_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert (1/5.0 + 3/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_combined_filtered_hit_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert (2/5.0 + 5/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, HITS_FEATURE_PREFIX, hit_level)])

@pytest.fixture(scope="module")
def sep_stat_gather():
  return StatGather.create()
