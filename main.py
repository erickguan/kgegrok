import torch
import data
import models
from estimate import train_and_validate
import logging
import torch.optim as optim
import visdom
import numpy as np
import sys
import argparse
from itertools import filterfalse
from utils import report_gpu_info

class Config(object):
    data_dir = "data/YAGO3-10"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 5
    negative_relation = 1
    batch_size = 100
    num_workers = 2
    entity_embedding_dimension = 50
    margin = 1.0
    epoches = 1000
    alpha = 0.001
    lr_decay = 0.96
    weight_decay = 0.96
    lambda_ = 0.001
    report_features = data.LinkPredictionStatistics.DEFAULT
    report_dimension = data.StatisticsDimension.DEFAULT
    # filename to resume
    resume = None
    # Introduce underministic behaviour but allow cudnn find best algoritm
    cudnn_benchmark = True
    logging_path = "logs"
    name = "TransE-YAGO3_10"
    enable_cuda = True
    # due to tile in the evaluation, it's reasonable to have less batch size
    evaluation_load_factor = 0.001
    plot_graph = True

    @classmethod
    def registered_options(cls):
        """Returns a iterator"""
        return filterfalse(lambda x: x.startswith('_') or type(cls.__dict__[x]) == classmethod, cls.__dict__)

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
            if v is not None:
                self.__dict__[k] = v


def cli(args):
    parser = argparse.ArgumentParser()
    for k in Config.registered_options():
        parser.add_argument("--{}".format(k), type=Config.option_type(k))

    parsed_args = parser.parse_args(args)
    config = Config(vars(parsed_args))
    config.enable_cuda = True if torch.cuda.is_available() and config.enable_cuda else False
    print(config.__dict__)
    input("Continue? [Ctrl-C] to stop.")

    np.random.seed(10000)
    torch.manual_seed(20000)
    torch.cuda.manual_seed_all(2192)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    drawer = visdom.Visdom(port=6006) if config.plot_graph else None
    model = train_and_validate(config, models.TransE, optim.Adam, drawer)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report_gpu_info()

    cli(sys.argv[1:])

