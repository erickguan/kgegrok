import torch
import data
from estimate import train_and_validate, test, train, interactive_prediction
import logging
import torch.optim as optim
import visdom
import numpy as np
import sys
import select
import argparse
from itertools import filterfalse
from utils import report_gpu_info, load_class_from_module, read_triple_translation
import stats
import importlib
import evaluation
import torch.multiprocessing as mp


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
    report_features = stats.LinkPredictionStatistics.DEFAULT
    report_dimension = stats.StatisticsDimension.DEFAULT

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
            if v is not None and k in self.registered_options():
                self.__dict__[k] = v

def cli_train(triple_source, config, model_class, optimizer_class, pool):
    drawer = stats.ReportDrawer(visdom.Visdom(port=6006), config) if config.plot_graph else None
    model = train_and_validate(triple_source, config, model_class, optimizer_class, pool, drawer)

def cli_test(triple_source, config, model_class, pool):
    assert config.resume is not None
    model = test(triple_source, config, model_class, pool)

def _get_and_validate_input(entities, relations):
    head = input("Head:")
    relation = input("Relation:")
    tail = input("Tail:")

    tokens = frozenset([head, relation, tail])
    if '?' not in tokens and len(tokens) == 3:
        raise RuntimeError("Which one to predict? Got ({}, {}, {})".format(head, relation, tail))
    if head not in entities or tail not in entities:
        raise RuntimeError("Head {} or tail {} not in entities tokens.".format(head, tail))
    if relation not in relations:
        raise RuntimeError("Relation {} not in entities tokens.".format(relation))
    yield head, relation, tail

def cli_demo_prediction(triple_source, config, model_class):
    """Iterative demo"""
    assert config.resume is not None
    config.enable_cuda = False
    entities, relations = read_triple_translation(config)
    generator = _get_and_validate_input(entities, relations)
    interactive_prediction(triple_source, entities, relations, config, model_class, generator)

def cli(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train')
    for k in Config.registered_options():
        parser.add_argument("--{}".format(k), type=Config.option_type(k))

    parsed_args = vars(parser.parse_args(args[1:]))
    config = Config(parsed_args)
    config.enable_cuda = True if torch.cuda.is_available() and config.enable_cuda else False
    print(config.__dict__)
    print("Continue? Starts in 10s. [Ctrl-C] to stop.")

    # let's save some seconds
    select.select([sys.stdin], [], [], 3)

    np.random.seed(10000)
    torch.manual_seed(20000)
    torch.cuda.manual_seed_all(2192)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    model_class = load_class_from_module(config.model, 'models', 'text_models')

    # TODO: DRY
    if parsed_args['moe'] in ['train', 'test']:
        ctx = mp.get_context('spawn')
        pool = evaluation.EvaluationProcessPool(config, triple_source, ctx)
        pool.start()

    # maybe roughly 10s now
    select.select([sys.stdin], [], [], 4)

    if parsed_args['mode'] == 'train':
        optimizer_class = load_class_from_module(config.optimizer, 'torch.optim')
        cli_train(triple_source, config, model_class, optimizer_class, pool)
    elif parsed_args['mode'] == 'test':
        cli_test(triple_source, config, model_class, pool)
    elif parsed_args['mode'] == 'demo_prediction':
        cli_demo_prediction(triple_source, config, model_class)

    # TODO: DRY
    if parsed_args['moe'] in ['train', 'test']:
        pool.stop()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report_gpu_info()

    cli(sys.argv)

