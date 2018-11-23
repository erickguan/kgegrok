import os.path
import json
import importlib
from pathlib import Path
from itertools import filterfalse

import torch

import kgekit.io
from kgexpr.stats.constants import *


def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))


def save_checkpoint(state,
                    filename='model_states/checkpoint.pth.tar',
                    postfix_num=None):
    path = "{}_{}".format(filename,
                          postfix_num) if postfix_num is not None else filename
    dirname = os.path.dirname(path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(config, model, optimizer=None):
    if config.resume:
        if os.path.isfile(config.resume):
            logging.info("loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("loaded checkpoint '{}' (epoch {})".format(
                config.resume, checkpoint['epoch']))
        else:
            logging.info("no checkpoint found at '{}'".format(config.resume))


def write_logging_data(raw_data, config):
    """writes the logging data."""
    with open(os.path.join(config.logging_path, config.name), 'w') as f:
        f.write(json.dumps(raw_data))


def load_class_from_module(class_name, *modules):
    for module in modules:
        mod = importlib.import_module(module)
        try:
            return getattr(mod, class_name)
        except:
            pass
    raise RuntimeError("Can't find the {} from {}".format(class_name, modules))


def read_triple_translation(config):
    translation_path = os.path.join(config.data_dir,
                                    config.translation_filename)
    entities, relations = kgekit.io.read_translation(translation_path)
    return dict(entities), dict(relations)


class Config(object):
    # Data
    data_dir = "data/YAGO3-10"
    triple_order = "hrt"
    triple_delimiter = ' '
    translation_filename = "translation.protobuf"

    # Data processing
    negative_entity = 5
    negative_relation = 1
    batch_size = 100
    num_workers = 2
    num_evaluation_workers = 2
    # due to tile in the evaluation, it's reasonable to have less batch size
    evaluation_load_factor = 0.0001

    # Model
    model = "TransE"
    optimizer = "Adam"

    # Model hyperparameters
    entity_embedding_dimension = 50
    margin = 1.0
    epoches = 1000
    alpha = 0.001
    lr_decay = 0.96
    weight_decay = 0.96
    lambda_ = 0.001

    # Stats
    report_features = LinkPredictionStatistics.DEFAULT
    report_dimension = StatisticsDimension.DEFAULT

    # Interactive response control
    report_num_prediction_interactively = 10

    # filename to resume
    resume = None
    # Introduce underministic behaviour but allow cudnn find best algoritm
    cudnn_benchmark = True
    enable_cuda = True

    # logging
    logging_path = "logs"
    name = "TransE-YAGO3_10"
    plot_graph = True

    @classmethod
    def registered_options(cls):
        """Returns a iterator"""
        return filterfalse(
            lambda x: x.startswith('_') or type(cls.__dict__[x]) == classmethod,
            cls.__dict__)

    @classmethod
    def option_type(cls, key):
        """Returns type class of given option"""
        return type(cls.__dict__[key])

    def __new__(cls, *args, **kwargs):
        instance = super(Config, cls).__new__(cls)
        for k in cls.registered_options():
            instance.__setattr__(k, cls.__dict__[k])
        return instance

    def __init__(self, args={}):
        for k, v in args.items():
            if v is not None and k in self.registered_options():
                self.__dict__[k] = v
