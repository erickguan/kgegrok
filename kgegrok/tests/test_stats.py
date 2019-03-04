import unittest
import pytest
from kgegrok import stats
from kgegrok.stats.constants import *


def test_rank_stats():
  res = stats.get_evaluation_statistics([1, 6, 11, 9, 12], [1, 5, 10, 9, 12],
                                        LinkPredictionStatistics.ALL)
  assert ((1.0 + 1 / 6.0 + 1 / 11.0 + 1 / 9.0 + 1 / 12.0) / 5.0 ==
    pytest.approx(res[MEAN_RECIPROCAL_RANK_FEATURE_KEY]))
  assert ((1.0 + 1 / 5.0 + 1 / 10.0 + 1 / 9.0 + 1 / 12.0) / 5.0 ==
      pytest.approx(res[MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY]))
  assert ((1 + 6 + 11 + 9 + 12) / 5.0 ==
    pytest.approx(res[MEAN_RANK_FEATURE_KEY]))
  assert ((1 + 5 + 10 + 9 + 12) / 5.0 ==
    pytest.approx(res[MEAN_FILTERED_RANK_FEATURE_KEY]))
  assert 1 / 5.0 == pytest.approx(res[HITS_1_FEATURE_KEY])
  assert 1 / 5.0 == pytest.approx(res[HITS_3_FEATURE_KEY])
  assert 1 / 5.0 == pytest.approx(res[HITS_5_FEATURE_KEY])
  assert 3 / 5.0 == pytest.approx(res[HITS_10_FEATURE_KEY])
  assert 1 / 5.0 == pytest.approx(res[HITS_1_FILTERED_FEATURE_KEY])
  assert 1 / 5.0 == pytest.approx(res[HITS_3_FILTERED_FEATURE_KEY])
  assert 2 / 5.0 == pytest.approx(res[HITS_5_FILTERED_FEATURE_KEY])
  assert 4 / 5.0 == pytest.approx(res[HITS_10_FILTERED_FEATURE_KEY])
