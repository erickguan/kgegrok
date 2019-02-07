"""Training function module."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import kgekit.data
import kgedata
from kgegrok import stats
from kgegrok import data
from kgegrok import evaluation
from kgegrok.data import constants
from kgegrok.utils import save_checkpoint, load_checkpoint, write_logging_data


def create_optimizer(optimizer_class, config, parameters):
    """return optimizer initialized with correct parameters."""
    if optimizer_class == optim.Adagrad:
        return optimizer_class(
            parameters,
            lr=config.alpha,
            lr_decay=config.lr_decay,
            weight_decay=config.weight_decay)
    elif optimizer_class == optim.Adadelta:
        return optimizer_class(parameters, lr=config.alpha)
    elif optimizer_class == optim.Adam:
        return optimizer_class(parameters, lr=config.alpha)
    else:
        return optimizer_class(parameters, lr=config.alpha)


def test(triple_source, config, model_class, evaluator):
    """Test config.resume model."""
    data_loader = data.create_dataloader(
        triple_source,
        config,
        collates_label=False,
        dataset_type=constants.DatasetType.TESTING)
    model = nn.DataParallel(model_class(triple_source, config))
    load_checkpoint(config, model)
    ranker = kgedata.Ranker(triple_source.train_set, triple_source.valid_set,
                           triple_source.test_set)

    if config.enable_cuda:
        model.cuda()

    logging.info('Testing starts')
    with torch.no_grad():
        results = evaluation.predict_links(model, triple_source, config, data_loader, evaluator)
    stats.report_prediction_result(config, results, drawer=None)

    return model

INDEX_OFFSET = 1

def train_and_validate(triple_source,
                       config,
                       model_class,
                       optimizer_class,
                       evaluator=None,
                       drawer=None,
                       enable_validation=True):
    """Train and validates the dataset."""
    # Data loaders have many processes. Here it's main process.
    data_loader = data.create_dataloader(triple_source, config,
                                         model_class.require_labels())
    if enable_validation:
        valid_data_loader = data.create_dataloader(
            triple_source,
            config,
            build_label=False,
            dataset_type=constants.DatasetType.VALIDATION)
    model = model_class(triple_source, config)
    if config.enable_cuda:
        model = nn.DataParallel(model)
        # has to be here because https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/7
        model.cuda()
    optimizer = create_optimizer(optimizer_class, config, model.parameters())
    load_checkpoint(config, model, optimizer)

    if drawer is not None:
        drawer.create_plot(stats.LOSS_FEATURE_KEY,
                           stats.gen_drawer_option(config, "Loss value"))
        if enable_validation:
            stats.prepare_plot_validation_result(drawer, config)

    for i_epoch in range(INDEX_OFFSET, config.epochs + INDEX_OFFSET, 1):
        model.train()
        logging.info('--------------------')
        logging.info('Training at epoch ' + str(i_epoch))
        logging.info('--------------------')

        loss_epoch = 0.0
        i_batch = 0
        for sample_batched in data_loader:
            logging.info('Training batch ' + str(i_batch + INDEX_OFFSET) + "/" +
                         str(len(data_loader.dataset)))

            loss = model.forward(sample_batched)
            loss_sum = loss.sum()
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            loss_epoch += float(loss_sum.item()) # avoid long-term memory usage
            i_batch += 1

        if drawer is not None:
            drawer.append(
                stats.LOSS_FEATURE_KEY,
                X=np.array([i_epoch], dtype='f'),
                Y=np.array([loss_epoch], dtype='f'))
        logging.info("Epoch " + str(i_epoch) + ": loss " + str(loss_epoch))

        if enable_validation:
            logging.info('Evaluation for epoch ' + str(i_epoch))
            with torch.no_grad():
                results = evaluation.predict_links(model, triple_source, config,
                                                   valid_data_loader, evaluator)
            stats.report_prediction_result(config, results, epoch=i_epoch, drawer=drawer)

        if config.save_per_epoch > 0 and i_epoch % config.save_per_epoch == 0:
            save_checkpoint({
                'epoch': i_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                            config,
                            postfix_num=i_epoch)

    if config.save_after_train:
        save_checkpoint({
            'epoch': config.epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
                        config,
                        postfix_num=config.epochs)

    if drawer is not None:
        write_logging_data(drawer.dump_raw_data(), config)

    return model


def train(triple_source, config, model_class, optimizer_class, drawer):
    """Train the dataset."""
    train_and_validate(
        triple_source,
        config,
        model_class,
        optimizer_class,
        drawer=drawer,
        enable_validation=False)

def interactive_prediction(triple_source, entities, relations, config,
                           model_class, generator):
    """prints prediction results according to config and generator inputs."""
    model = model_class(triple_source, config)
    load_checkpoint(config, model)
    model.eval()

    logging.info('Interactive prediction starts')
    with torch.no_grad():
        for head, relation, tail in generator:
            logging.info('--------------------')
            logging.info('Prediction input ({}, {}, {})'.format(
                head, relation, tail))
            logging.info('--------------------')

            # determines which element to predict
            batch, prediction_type, triple_index = data.sieve_and_expand_triple(
                triple_source, entities, relations, head, relation, tail)
            batch = data.convert_triple_tuple_to_torch(batch, config)
            with torch.no_grad():
                predicted = model.forward(batch).cpu()

            logging.info('Predicting {} for ({}, {}, {})'.format(
                repr(prediction_type), head, relation, tail))
            prediction_list = evaluation.evaluate_single_triple(
                predicted, prediction_type, triple_index, config, entities,
                relations)
            logging.info('Top {} predicted elements are:'.format(
                rank, filtered_rank, len(prediction_list)))
            for idx, prediction in prediction_list:
                logging.info('{}: {}'.format(idx, prediction))
