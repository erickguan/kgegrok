"""Training function module."""

import data
import kgekit.data
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import save_checkpoint, load_checkpoint, write_logging_data
from stats import evaulate_prediction, report_prediction_result, prepare_plot_validation_result, gen_drawer_key


def create_optimizer(optimizer_class, config, parameters):
    if optimizer_class == optim.Adagrad:
        return optimizer_class(parameters, lr=config.alpha, lr_decay=config.lr_decay, weight_decay=config.weight_decay)
    elif optimizer_class == optim.Adadelta:
        return optimizer_class(parameters, lr=config.alpha)
    elif optimizer_class == optim.Adam:
        return optimizer_class(parameters, lr=config.alpha)
    else:
        return optimizer_class(parameters, lr=config.alpha)


def train_and_validate(config, model_class, optimizer_class, drawer=None):
    # Data loaders have many processes. Here it's a main program.
    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    data_loader = data.create_dataloader(triple_source, config)
    valid_data_loader = data.create_dataloader(triple_source, config, data.DatasetType.VALIDATION)
    model = nn.DataParallel(model_class(triple_source, config))
    optimizer = create_optimizer(optimizer_class, config, model.parameters())
    load_checkpoint(model, optimizer, config)
    ranker = kgekit.Ranker(triple_source.train_set, triple_source.valid_set, triple_source.test_set)

    if torch.cuda.is_available():
        model.cuda()

    if drawer != None:
        loss_values_drawer = drawer.line(
            X=np.array([0.], dtype='f'),
            Y=np.array([0.], dtype='f'),
            opts=gen_drawer_key(config, "Loss value"))
        validation_results_drawer = prepare_plot_validation_result(drawer, config)

    INDEX_OFFSET = 1
    for i_epoch in range(INDEX_OFFSET, config.epoches+INDEX_OFFSET, 1):
        model.train()
        logging.info('--------------------')
        logging.info('Training at epoch ' + str(i_epoch))
        logging.info('--------------------')

        loss_epoch = 0.0
        for i_batch, sample_batched in enumerate(data_loader):
            logging.info('Training batch ' + str(i_batch+INDEX_OFFSET) + "/" + str(len(data_loader)))
            batch, negative_batch = sample_batched
            batch = data.convert_triple_tuple_to_torch(data.get_triples_from_batch(batch), config)
            negative_batch = data.convert_triple_tuple_to_torch(data.get_negative_samples_from_batch(negative_batch), config)
            loss = model.forward(batch, negative_batch)
            loss_sum = loss.sum()
            loss_sum.backward()
            optimizer.step()
            loss_epoch += loss_sum.data[0]

        drawer.line(X=np.array([i_epoch], dtype='f'), Y=np.array([loss_epoch], dtype='f'), win=loss_values_drawer, update='append')
        logging.info("Epoch " + str(i_epoch) + ": loss " + str(loss_epoch))

        logging.info('Evaluation for epoch ' + str(i_epoch))
        result = evaulate_prediction(model, triple_source, config, ranker, valid_data_loader)
        report_prediction_result(i_epoch, result, config, drawer, validation_results_drawer, triple_source)
        save_checkpoint({
            'epoch': i_epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, "model_states/" + config.name + "/checkpoint.pth.tar", postfix_num=i_epoch)
        drawing_wins = list(validation_results_drawer.values()) + [loss_values_drawer]
    write_logging_data(drawer, drawing_wins, config)

    return model
