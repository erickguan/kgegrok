import pprint
import data
import kgekit
import numpy as np
import logging

def _evaluate_predict_element(model, config, triple_index, num_expands, element_type, rank_fn, ranks_list, filtered_ranks_list):
    """Evaluation a single triple with expanded sets."""
    batch = data.expand_triple_to_sets(kgekit.data.unpack(triple_index), num_expands, element_type)
    batch = data.convert_triple_tuple_to_torch(batch, config)
    logging.debug(element_type)
    logging.debug("Batch len: " + str(len(batch)) + "; batch sample: " + str(batch[0]))
    predicted_batch = model.forward(batch).cpu()
    logging.debug("Predicted batch len" + str(len(predicted_batch)) + "; batch sample: " + str(predicted_batch[0]))
    rank, filtered_rank = rank_fn(predicted_batch.data.numpy(), triple_index)
    logging.debug("Rank :" + str(rank) + "; Filtered rank length :" + str(filtered_rank))
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

def _evaluate_prediction_view(result_view, triple_index, rank_fn, ranks_list, filtered_ranks_list):
    """Evaluation on a view of batch."""
    rank, filtered_rank = rank_fn(result_view, triple_index)
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

def gen_drawer_key(config, title=None):
    if title is not None:
        title = "{}/{}".format(config.name, title)
    return dict(fillarea=True, title=title)

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
        sampled, batch, splits = sample_batched
        sampled = data.convert_triple_tuple_to_torch(data.get_triples_from_batch(sampled), config)
        predicted_batch = model.forward(sampled).cpu().data.numpy()

        for triple_index, split in zip(batch, splits):
            if split[0] != split[1] and split[1] != split[2]:
                _evaluate_prediction_view(predicted_batch[split[0]:split[1]], triple_index, ranker.rankHead, head_ranks, filtered_head_ranks)
                _evaluate_prediction_view(predicted_batch[split[1]:split[2]], triple_index, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if split[2] != split[3]:
                _evaluate_prediction_view(predicted_batch[split[2]:split[3]], triple_index, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

def evaulate_prediction_np_collate(model, triple_source, config, ranker, data_loader):
    """use with NumpyCollate."""
    model.eval()

    head_ranks = []
    filtered_head_ranks = []
    tail_ranks = []
    filtered_tail_ranks = []
    relation_ranks = []
    filtered_relation_ranks = []

    for i_batch, sample_batched in enumerate(data_loader):
        # sample_batched is a list of triple. triple has shape (1, 3). We need to tile it for the test.
        for triple in sample_batched:
            triple_index = kgekit.TripleIndex(*triple[0, :])

            if (config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY) or (config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY):
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.report_dimension & data.StatisticsDimension.RELATION:
                _evaluate_predict_element(model, config, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

def _add_rank_plot_maker(config, drawers, drawer, key, *add):
    if not any(add): return
    drawers[key] = drawer.line(
            X=np.array([0.], dtype='f'),
            Y=np.array([0.], dtype='f'),
            opts=gen_drawer_key(config, key))

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
