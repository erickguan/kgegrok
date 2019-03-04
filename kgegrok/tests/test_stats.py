import unittest
import pytest
from kgegrok import stats
from kgegrok.stats.constants import *
from kgegrok.data import constants, statstools

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
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, FILTERED_HITS_FEATURE_PREFIX, hit_level)])

def test_tail_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.TAIL_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert 2 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_tail_filtered_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.TAIL_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, FILTERED_HITS_FEATURE_PREFIX, hit_level)])

def test_rel_hit_stat_gather(rank_results):
  hit_level = 9
  gather = stats.StatGather([stats.ElementHitStatTool(constants.RELATION_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_rel_filtered_hit_stat_gather(rank_results):
  hit_level = 5
  gather = stats.StatGather([stats.ElementHitStatTool(constants.RELATION_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, FILTERED_HITS_FEATURE_PREFIX, hit_level)])

def test_combined_hit_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, hit_level, filtered=False)])
  results = gather(rank_results)
  assert (1/5.0 + 3/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, HITS_FEATURE_PREFIX, hit_level)])

def test_combined_filtered_hit_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, hit_level, filtered=True)])
  results = gather(rank_results)
  assert (2/5.0 + 5/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, FILTERED_HITS_FEATURE_PREFIX, hit_level)])

def test_head_mean_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanRankStatTool(constants.HEAD_KEY, filtered=False)])
  results = gather(rank_results)
  assert (9+3+7+11+10) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_RANK_FEATURE_KEY)])

def test_head_filtered_mean_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanRankStatTool(constants.HEAD_KEY, filtered=True)])
  results = gather(rank_results)
  assert (9+2+6+8+9) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])

def test_tail_mean_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanRankStatTool(constants.TAIL_KEY, filtered=False)])
  results = gather(rank_results)
  assert (7+6+8+5+4) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_RANK_FEATURE_KEY)])

def test_tail_filtered_mean_rank_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.ElementMeanRankStatTool(constants.TAIL_KEY, filtered=True)])
  results = gather(rank_results)
  assert (6+6+3+5+1) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])

def test_combined_mean_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.CombinedEntityMeanRankStatTool(constants.ENTITY_KEY, filtered=False)])
  results = gather(rank_results)
  assert ((9+3+7+11+10)/5.0 + (7+6+8+5+4)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_RANK_FEATURE_KEY)])

def test_combined_filtered_mean_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.CombinedEntityMeanRankStatTool(constants.ENTITY_KEY, filtered=True)])
  results = gather(rank_results)
  assert ((9+2+6+8+9)/5.0 + (6+6+3+5+1)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])

def test_head_mean_reciprocal_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanReciprocalRankStatTool(constants.HEAD_KEY, filtered=False)])
  results = gather(rank_results)
  assert (1/9.0+1/3.0+1/7.0+1/11.0+1/10.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])

def test_head_filtered_mean_reciprocal_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanReciprocalRankStatTool(constants.HEAD_KEY, filtered=True)])
  results = gather(rank_results)
  assert (1/9.0+1/2.0+1/6.0+1/8.0+1/9.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])

def test_tail_mean_reciprocal_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.ElementMeanReciprocalRankStatTool(constants.TAIL_KEY, filtered=False)])
  results = gather(rank_results)
  assert (1/7.0+1/6.0+1/8.0+1/5.0+1/4.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])

def test_tail_filtered_mean_reciprocal_rank_stat_gather(rank_results):
  hit_level = 6
  gather = stats.StatGather([stats.ElementMeanReciprocalRankStatTool(constants.TAIL_KEY, filtered=True)])
  results = gather(rank_results)
  assert (1/6.0+1/6.0+1/3.0+1/5.0+1/1.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])

def test_combined_mean_reciprocal_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.CombinedEntityMeanReciprocalRankStatTool(constants.ENTITY_KEY, filtered=False)])
  results = gather(rank_results)
  assert ((1/9.0+1/3.0+1/7.0+1/11.0+1/10.0)/5.0 + (1/7.0+1/6.0+1/8.0+1/5.0+1/4.0)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])

def test_combined_filtered_mean_reciprocal_rank_stat_gather(rank_results):
  gather = stats.StatGather([stats.CombinedEntityMeanReciprocalRankStatTool(constants.ENTITY_KEY, filtered=True)])
  results = gather(rank_results)
  assert ((1/9.0+1/2.0+1/6.0+1/8.0+1/9.0)/5.0 + (1/6.0+1/6.0+1/3.0+1/5.0+1/1.0)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])

@pytest.fixture(scope="module")
def sep_stat_gather():
  return stats.StatGather([
    stats.ElementHitStatTool(constants.HEAD_KEY, 5, filtered=False),
    stats.ElementHitStatTool(constants.HEAD_KEY, 8, filtered=True),
    stats.ElementHitStatTool(constants.TAIL_KEY, 5, filtered=False),
    stats.ElementHitStatTool(constants.TAIL_KEY, 5, filtered=True),
    stats.ElementHitStatTool(constants.RELATION_KEY, 9, filtered=False),
    stats.ElementHitStatTool(constants.RELATION_KEY, 5, filtered=True),
    stats.ElementMeanRankStatTool(constants.HEAD_KEY, filtered=False),
    stats.ElementMeanRankStatTool(constants.HEAD_KEY, filtered=True),
    stats.ElementMeanRankStatTool(constants.TAIL_KEY, filtered=False),
    stats.ElementMeanRankStatTool(constants.TAIL_KEY, filtered=True),
    stats.ElementMeanReciprocalRankStatTool(constants.HEAD_KEY, filtered=False),
    stats.ElementMeanReciprocalRankStatTool(constants.HEAD_KEY, filtered=True),
    stats.ElementMeanReciprocalRankStatTool(constants.TAIL_KEY, filtered=False),
    stats.ElementMeanReciprocalRankStatTool(constants.TAIL_KEY, filtered=True),
  ])

def test_sep_stat_gather(sep_stat_gather, rank_results):
  results = sep_stat_gather(rank_results)
  assert 1 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, HITS_FEATURE_PREFIX, 5)])
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, FILTERED_HITS_FEATURE_PREFIX, 8)])
  assert 2 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, HITS_FEATURE_PREFIX, 5)])
  assert 3 / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, FILTERED_HITS_FEATURE_PREFIX, 5)])
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, HITS_FEATURE_PREFIX, 9)])
  assert 2 / 3.0 == pytest.approx(results[stats.StatTool.gen_key(constants.RELATION_KEY, FILTERED_HITS_FEATURE_PREFIX, 5)])
  assert (9+3+7+11+10) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_RANK_FEATURE_KEY)])
  assert (9+2+6+8+9) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])
  assert (7+6+8+5+4) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_RANK_FEATURE_KEY)])
  assert (6+6+3+5+1) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])
  assert (1/9.0+1/3.0+1/7.0+1/11.0+1/10.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])
  assert (1/9.0+1/2.0+1/6.0+1/8.0+1/9.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.HEAD_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])
  assert (1/7.0+1/6.0+1/8.0+1/5.0+1/4.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])
  assert (1/6.0+1/6.0+1/3.0+1/5.0+1/1.0) / 5.0 == pytest.approx(results[stats.StatTool.gen_key(constants.TAIL_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])

@pytest.fixture(scope="module")
def combined_stat_gather():
  return stats.StatGather([
    stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, 6, filtered=False),
    stats.CombinedEntityHitStatTool(constants.ENTITY_KEY, 6, filtered=True),
    stats.CombinedEntityMeanRankStatTool(constants.ENTITY_KEY, filtered=False),
    stats.CombinedEntityMeanRankStatTool(constants.ENTITY_KEY, filtered=True),
    stats.CombinedEntityMeanReciprocalRankStatTool(constants.ENTITY_KEY, filtered=False),
    stats.CombinedEntityMeanReciprocalRankStatTool(constants.ENTITY_KEY, filtered=True)
  ])

def test_combined_stat_gather(combined_stat_gather, rank_results):
  results = combined_stat_gather(rank_results)
  hit_level = 6
  assert (1/5.0 + 3/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, HITS_FEATURE_PREFIX, hit_level)])
  assert (2/5.0 + 5/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, FILTERED_HITS_FEATURE_PREFIX, hit_level)])
  assert ((9+3+7+11+10)/5.0 + (7+6+8+5+4)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_RANK_FEATURE_KEY)])
  assert ((9+2+6+8+9)/5.0 + (6+6+3+5+1)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_FILTERED_RANK_FEATURE_KEY)])
  assert ((1/9.0+1/3.0+1/7.0+1/11.0+1/10.0)/5.0 + (1/7.0+1/6.0+1/8.0+1/5.0+1/4.0)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_RECIPROCAL_RANK_FEATURE_KEY)])
  assert ((1/9.0+1/2.0+1/6.0+1/8.0+1/9.0)/5.0 + (1/6.0+1/6.0+1/3.0+1/5.0+1/1.0)/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY)])

def test_stat_gather_after_gather_hook(combined_stat_gather, rank_results):
  def after_gather_hook(results, ranks, epoch):
    assert ranks == rank_results
    assert epoch == 1
    assert (1/5.0 + 3/5.0) / 2.0 == pytest.approx(results[stats.StatTool.gen_key(constants.ENTITY_KEY, HITS_FEATURE_PREFIX, 6)])
  combined_stat_gather.add_after_gather(after_gather_hook)
  combined_stat_gather(rank_results, 1)
