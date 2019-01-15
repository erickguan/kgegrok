from enum import Flag, auto
from kgegrok.data import constants
from kgegrok.data import statstools

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
LOSS_FEATURE_KEY = 'loss'


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
    DEFAULT = HITS_FILTERED | MEAN_FILTERED_RANK
    ALL = ALL_RANKS | ALL_HITS


class StatisticsDimension(Flag):
    SEPERATE_ENTITY = auto()
    COMBINED_ENTITY = auto()
    RELATION = auto()
    DEFAULT = COMBINED_ENTITY | RELATION
    ALL = SEPERATE_ENTITY | RELATION


DRAWING_KEY_AND_CONDITION = {
    StatisticsDimension.COMBINED_ENTITY: [
        (MEAN_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_RANK),
        (MEAN_FILTERED_RANK_FEATURE_KEY,
         LinkPredictionStatistics.MEAN_FILTERED_RANK),
        (MEAN_RECIPROCAL_RANK_FEATURE_KEY,
         LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
        (MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY,
         LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
        (HITS_1_FEATURE_KEY, LinkPredictionStatistics.HITS_1),
        (HITS_3_FEATURE_KEY, LinkPredictionStatistics.HITS_3),
        (HITS_5_FEATURE_KEY, LinkPredictionStatistics.HITS_5),
        (HITS_10_FEATURE_KEY, LinkPredictionStatistics.HITS_10),
        (HITS_1_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_1_FILTERED),
        (HITS_3_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_3_FILTERED),
        (HITS_5_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_5_FILTERED),
        (HITS_10_FILTERED_FEATURE_KEY,
         LinkPredictionStatistics.HITS_10_FILTERED),
    ],
    StatisticsDimension.SEPERATE_ENTITY: [
        (statstools.dict_key_gen(constants.HEAD_KEY, MEAN_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RANK),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 MEAN_FILTERED_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RANK),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 MEAN_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.HEAD_KEY, HITS_1_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1),
        (statstools.dict_key_gen(constants.HEAD_KEY, HITS_3_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3),
        (statstools.dict_key_gen(constants.HEAD_KEY, HITS_5_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5),
        (statstools.dict_key_gen(constants.HEAD_KEY, HITS_10_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 HITS_1_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1_FILTERED),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 HITS_3_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3_FILTERED),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 HITS_5_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5_FILTERED),
        (statstools.dict_key_gen(constants.HEAD_KEY,
                                 HITS_10_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10_FILTERED),
        (statstools.dict_key_gen(constants.TAIL_KEY, MEAN_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RANK),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 MEAN_FILTERED_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RANK),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 MEAN_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.TAIL_KEY, HITS_1_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1),
        (statstools.dict_key_gen(constants.TAIL_KEY, HITS_3_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3),
        (statstools.dict_key_gen(constants.TAIL_KEY, HITS_5_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5),
        (statstools.dict_key_gen(constants.TAIL_KEY, HITS_10_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 HITS_1_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1_FILTERED),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 HITS_3_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3_FILTERED),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 HITS_5_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5_FILTERED),
        (statstools.dict_key_gen(constants.TAIL_KEY,
                                 HITS_10_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10_FILTERED),
    ],
    StatisticsDimension.RELATION: [
        (statstools.dict_key_gen(constants.RELATION_KEY, MEAN_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RANK),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 MEAN_FILTERED_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RANK),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 MEAN_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY),
         LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
        (statstools.dict_key_gen(constants.RELATION_KEY, HITS_1_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1),
        (statstools.dict_key_gen(constants.RELATION_KEY, HITS_3_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3),
        (statstools.dict_key_gen(constants.RELATION_KEY, HITS_5_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5),
        (statstools.dict_key_gen(constants.RELATION_KEY, HITS_10_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 HITS_1_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_1_FILTERED),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 HITS_3_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_3_FILTERED),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 HITS_5_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_5_FILTERED),
        (statstools.dict_key_gen(constants.RELATION_KEY,
                                 HITS_10_FILTERED_FEATURE_KEY),
         LinkPredictionStatistics.HITS_10_FILTERED),
    ]
}
