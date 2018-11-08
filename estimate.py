"""Training function module."""

import data
import kgekit.data
import logging
import torch
import pprint
import torch.nn as nn
import torch.optim as optim
import numpy as np


def _evaluate_predict_element(model, triple_index, num_expands, element_type, rank_fn, ranks_list, filtered_ranks_list):
    batch = data.expand_triple_to_sets(kgekit.data.unpack(triple_index), num_expands, element_type)
    batch = data.convert_triple_tuple_to_torch(batch)
    logging.debug(element_type)
    logging.debug("Batch len: " + str(len(batch)) + "; batch sample: " + str(batch[0]))
    predicted_batch = model.forward(batch).cpu()
    logging.debug("Predicted batch len" + str(len(predicted_batch)) + "; batch sample: " + str(predicted_batch[0]))
    rank, filtered_rank = rank_fn(predicted_batch.data.numpy(), triple_index)
    logging.debug("Rank :" + str(rank) + "; Filtered rank length :" + str(filtered_rank))
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

def _report_prediction_element(element):
    pprint.pprint(element)

def _common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def _append_drawer(epoch, drawer, drawers, result, prefix_key=None):
    for key, value in result.items():
        drawer_key = data.dict_key_gen(prefix_key, key) if prefix_key is not None else key
        drawer.line(X=np.array([epoch], dtype='f'), Y=np.array([value], dtype='f'), win=drawers[drawer_key], update='append')

def report_prediction_result(epoch, result, config, drawer, results_drawer):
    heads, tails, relations = result

    if config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY:
        head_result = data.get_rank_statistics(*heads, config.report_features)
        tail_result = data.get_rank_statistics(*tails, config.report_features)
        _report_prediction_element(head_result)
        _report_prediction_element(tail_result)
        _append_drawer(epoch, drawer, results_drawer, head_result, data.HEAD_KEY)
        _append_drawer(epoch, drawer, results_drawer, head_result, data.TAIL_KEY)

    elif config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY:
        combined = {k: (h + t) / 2.0 for k, h, t in _common_entries(data.get_rank_statistics(*heads, config.report_features), data.get_rank_statistics(*tails, config.report_features))}
        _report_prediction_element(combined)
        _append_drawer(epoch, drawer, results_drawer, combined)

    if config.report_dimension & data.StatisticsDimension.RELATION:
        relation_result = data.get_rank_statistics(*relations, config.report_features)
        _report_prediction_element(rellation_result)
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

            if config.test_head:
                _evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
            if config.test_tail:
                _evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.test_relation:
                _evaluate_predict_element(model, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

def create_optimizer(optimizer_class, config, parameters):
    if optimizer_class == optim.Adagrad:
        return optimizer_class(parameters, lr=config.alpha, lr_decay=self.lr_decay, weight_decay=self.weight_decay)
    elif optimizer_class == optim.Adadelta:
        return optimizer_class(parameters, lr=config.alpha)
    elif optimizer_class == optim.Adam:
        return optimizer_class(parameters, lr=config.alpha)
    else:
        return optimizer_class(parameters, lr=config.alpha)


def _add_rank_plot_maker(config, drawers, drawer, key, *add):
    if not any(add): return
    drawers[key] = drawer.line(
            X=np.array([0.], dtype='f'),
            Y=np.array([0.], dtype='f'),
            opts=data.gen_drawer_key(config, key))

def _prepare_plot_validation_result(drawer, config):
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

def train_and_validate(config, model_class, optimizer_class, drawer=None):
    # Data loaders have many processes. Here it's a main program.
    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    data_loader = data.create_dataloader(triple_source, config)
    valid_data_loader = data.create_dataloader(triple_source, config, data.DatasetType.VALIDATION)
    model = nn.DataParallel(model_class(triple_source, config))
    optimizer = create_optimizer(optimizer_class, config, model.parameters())
    ranker = kgekit.Ranker(triple_source.train_set, triple_source.valid_set, triple_source.test_set)

    if torch.cuda.is_available():
        model.cuda()

    if drawer != None:
        loss_values_drawer = drawer.line(
            X=np.array([0.], dtype='f'),
            Y=np.array([0.], dtype='f'),
            opts=data.gen_drawer_key(config, "Loss value"))
        validation_results_drawer = _prepare_plot_validation_result(drawer, config)

    for i_epoch in range(config.epoches):
        model.train()
        logging.info('--------------------')
        logging.info('Training at epoch ' + str(i_epoch))
        logging.info('--------------------')

        loss_epoch = 0.0
        for i_batch, sample_batched in enumerate(data_loader):
            logging.info('Training batch ' + str(i_batch) + "/" + str(len(data_loader)))
            batch, negative_batch = sample_batched
            batch = data.convert_triple_tuple_to_torch(data.get_triples_from_batch(batch))
            negative_batch = data.convert_triple_tuple_to_torch(data.get_negative_samples_from_batch(negative_batch))
            loss = model.forward(batch, negative_batch)
            loss.sum().backward()
            optimizer.step()
            loss_epoch += loss.data[0]

        drawer.line(X=np.array([i_epoch], dtype='f'), Y=np.array([loss_epoch], dtype='f'), win=loss_values_drawer,update='append')
        logging.info("Epoch " + str(i_epoch) + ": loss " + str(loss_epoch))

        logging.info('Evaluation for epoch ' + str(i_epoch))
        result = evaulate_prediction(model, triple_source, config, ranker, valid_data_loader)
        report_prediction_result(i_epoch, result, config, drawer, validation_results_drawer)

    return model
