import pprint
import data
import kgekit
import numpy as np
from estimate import evaluate_predict_element


def _report_prediction_element(element):
    pprint.pprint(element)

def _common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def _append_drawer(epoch, drawer, drawers, result, prefix_key=None):
    for key, value in result.items():
        drawer_key = data.dict_key_gen(prefix_key, key) if prefix_key is not None else key
        drawer.line(X=np.array([epoch], dtype='f'), Y=np.array([value], dtype='f'), win=drawers[drawer_key], update='append')

def report_prediction_result(epoch, result, config, drawer, results_drawer, triple_source):
    heads, tails, relations = result

    if config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY:
        head_result = data.get_rank_statistics(*heads, config.report_features, triple_source.num_entity)
        tail_result = data.get_rank_statistics(*tails, config.report_features, triple_source.num_entity)
        _report_prediction_element(head_result)
        _report_prediction_element(tail_result)
        _append_drawer(epoch, drawer, results_drawer, head_result, data.HEAD_KEY)
        _append_drawer(epoch, drawer, results_drawer, head_result, data.TAIL_KEY)

    elif config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY:
        combined = {k: (h + t) / 2.0 for k, h, t in _common_entries(data.get_rank_statistics(*heads, config.report_features, triple_source.num_entity), data.get_rank_statistics(*tails, config.report_features, triple_source.num_entity))}
        _report_prediction_element(combined)
        _append_drawer(epoch, drawer, results_drawer, combined)

    if config.report_dimension & data.StatisticsDimension.RELATION:
        relation_result = data.get_rank_statistics(*relations, config.report_features, triple_source.num_entity)
        _report_prediction_element(relation_result)
        _append_drawer(epoch, drawer, results_drawer, relation_result, data.RELATION_KEY)

def evaulate_prediction(model, triple_source, config, ranker, data_loader):
    model.eval()

    head_ranks = []
    filtered_head_ranks = []
    tail_ranks = []
    filtered_tail_ranks = []
    relation_ranks = []
    filtered_relation_ranks = []

    for i_batch, sample_batched in enumerate(data_loader):
        # triple has shape (1, 3). We need to tile it for the test.
        for triple in sample_batched:
            triple_index = kgekit.TripleIndex(*triple[0, :])

            if (config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY) or (config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY):
                evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
                evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.report_dimension & data.StatisticsDimension.RELATION:
                evaluate_predict_element(model, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

def _add_rank_plot_maker(config, drawers, drawer, key, *add):
    if not any(add): return
    drawers[key] = drawer.line(
            X=np.array([0.], dtype='f'),
            Y=np.array([0.], dtype='f'),
            opts=data.gen_drawer_key(config, key))

def prepare_plot_validation_result(drawer, config):
    drawers = {}
    if config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY:
        _add_rank_plot_maker(config, drawers, drawer, data.MEAN_RANK_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.MEAN_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.MEAN_FILTERED_RANK_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_1_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_1)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_3_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_3)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_5_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_5)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_10_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_10)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_1_FILTERED_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_1_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_3_FILTERED_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_3_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_5_FILTERED_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_5_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.HITS_10_FILTERED_FEATURE_KEY, config.report_features & data.LinkPredictionStatistics.HITS_10_FILTERED)
    elif config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY:
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.MEAN_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_1_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_3_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_5_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_10_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_1_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_3_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_5_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.HEAD_KEY, data.HITS_10_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10_FILTERED)

        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.MEAN_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_1_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_3_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_5_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_10_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_1_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_3_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_5_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.TAIL_KEY, data.HITS_10_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10_FILTERED)

    if config.report_dimension & data.StatisticsDimension.RELATION:
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.MEAN_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_1_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_3_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_5_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_10_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_1_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_1_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_3_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_3_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_5_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_5_FILTERED)
        _add_rank_plot_maker(config, drawers, drawer, data.dict_key_gen(data.RELATION_KEY, data.HITS_10_FILTERED_FEATURE_KEY), config.report_features & data.LinkPredictionStatistics.HITS_10_FILTERED)

    return drawers
