"""Training function module."""

import data
import kgekit.data
import logging
import torch
import pprint
import torch.nn as nn


def _evaluate_predict_element(model, triple_index, num_expands, element_type, rank_fn, ranks_list, filtered_ranks_list):
    batch = data.expand_triple_to_sets(kgekit.data.unpack(triple_index), num_expands, element_type)
    batch = data.convert_triple_tuple_to_torch(batch)
    logging.debug(element_type)
    logging.debug("Batch len: " + str(len(batch)) + "; batch sample: " + str(batch[0]))
    predicted_batch = model.predict(batch)
    logging.debug("Predicted batch len" + str(len(batch)) + "; batch sample: " + str(predicted_batch[0]))
    rank, filtered_rank = rank_fn(predicted_batch.data.numpy(), triple_index)
    logging.debug("Rank :" + str(rank) + "; Filtered rank length :" + str(filtered_rank))
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

def _report_prediction_element(element, features):
    rank, filtered_rank = element
    if len(rank) == 0:
        return
    pprint.pprint(data.get_rank_statistics(rank, features))
    pprint.pprint(data.get_rank_statistics(filtered_rank, features))


def report_prediction_result(result, features):
    heads, tails, relations = result
    _report_prediction_element(heads, features)
    _report_prediction_element(tails, features)
    _report_prediction_element(relations, features)

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
                logging.debug('validate head prediction')
                _evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
            if config.test_tail:
                logging.debug('validate tail prediction')
                _evaluate_predict_element(model, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.test_relation:
                logging.debug('validate relation prediction')
                _evaluate_predict_element(model, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

def train_and_validate(config, model_klass):
    # Data loaders have many processes. Here it's a main program.
    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    data_loader = data.create_dataloader(triple_source, config)
    valid_data_loader = data.create_dataloader(triple_source, config, data.DatasetType.VALIDATION)
    model = nn.DataParallel(model_klass(triple_source, config))
    ranker = kgekit.Ranker(triple_source.train_set, triple_source.valid_set, triple_source.test_set)

    if torch.cuda.is_available():
        model.cuda()

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
            loss_epoch += loss[0].item()

        logging.info("Epoch " + str(i_epoch) + ": loss " + str(loss_epoch))

        logging.info('Evaluation for epoch ' + str(i_epoch))
        result = evaulate_prediction(model, triple_source, config, ranker, valid_data_loader)
        report_prediction_result(result, data.LinkPredictionStatistics.DEFAULT)

    return model
