import logging
import numpy as np
import sys
import select
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from kgegrok import data
from kgegrok import estimate
from kgegrok import stats
from kgegrok import evaluation
from kgegrok import utils
from kgegrok.stats import create_drawer

def cli_profile(triple_source, config, model_class, optimizer_class):
    with torch.autograd.profiler.profile() as prof:
        model = estimate.train(triple_source, config, model_class,
                        optimizer_class, drawer=create_drawer(config))
    print(prof)

def cli_train(triple_source, config, model_class, optimizer_class):
    model = estimate.train(triple_source, config, model_class,
                           optimizer_class, drawer=create_drawer(config))

def cli_train_and_validate(triple_source, config, model_class, optimizer_class, validation_evaluator):
    model = estimate.train_and_validate(triple_source, config, model_class,
                                        optimizer_class, validation_evaluator, drawer=create_drawer(config))

def cli_test(triple_source, config, model_class, validation_evaluator):
    assert config.resume is not None and len(config.resume) > 0
    model = estimate.test(triple_source, config, model_class, validation_evaluator)


def _get_and_validate_input(entities, relations):
    head = input("Head:")
    relation = input("Relation:")
    tail = input("Tail:")

    tokens = frozenset([head, relation, tail])
    if '?' not in tokens and len(tokens) == 3:
        raise RuntimeError("Which one to predict? Got ({}, {}, {})".format(
            head, relation, tail))
    if head not in entities or tail not in entities:
        raise RuntimeError("Head {} or tail {} not in entities tokens.".format(
            head, tail))
    if relation not in relations:
        raise RuntimeError(
            "Relation {} not in entities tokens.".format(relation))
    yield head, relation, tail


def cli_demo_prediction(triple_source, config, model_class):
    """Iterative demo"""
    assert config.resume is not None and len(config.resume) > 0
    config.enable_cuda = False
    entities, relations = utils.read_triple_translation(config)
    generator = _get_and_validate_input(entities, relations)
    estimate.interactive_prediction(triple_source, entities, relations, config,
                                    model_class, generator)


def cli_config_and_parse_args(args):
    parser = argparse.ArgumentParser()
    for k in utils.Config.registered_options():
        cfg_type = utils.Config.option_type(k)
        if cfg_type == bool:
            parser.add_argument("--{}".format(k), type=utils.str2bool)
        else:
            parser.add_argument("--{}".format(k), type=utils.Config.option_type(k))

    parsed_args = vars(parser.parse_args(args[1:]))
    return utils.build_config_with_dict(parsed_args)


def cli(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    utils.report_gpu_info()

    config = cli_config_and_parse_args(args)
    print(config.__dict__)
    print("Continue? Starts in 10s. [Ctrl-C] to stop.")

    # let's save some seconds
    select.select([sys.stdin], [], [], 3)

    utils.seed_modules(
        config,
        numpy_seed=10000,
        torch_seed=20000,
        torcu_cuda_seed_all=2192,
        cuda_deterministic=True,
        kgegrok_base_seed=30000,
        cuda_benchmark=config.cudnn_benchmark)

    triple_source = data.TripleSource(config.data_dir, config.triple_order,
                                      config.triple_delimiter)
    model_class = utils.load_class_from_module(config.model, 'kgegrok.models',
                                         'kgegrok.text_models')

    evaluator = evaluation.ParallelEvaluator(config, triple_source)
    # maybe roughly 10s now
    select.select([sys.stdin], [], [], 4)

    if config.mode == 'train':
        optimizer_class = utils.load_class_from_module(config.optimizer,
                                                'torch.optim')
        cli_train(triple_source, config, model_class, optimizer_class)
    elif config.mode == 'train_validate':
        optimizer_class = utils.load_class_from_module(config.optimizer,
                                                'torch.optim')
        cli_train_and_validate(triple_source, config, model_class, optimizer_class, evaluator)
    elif config.mode == 'test':
        cli_test(triple_source, config, model_class, evaluator)
    elif config.mode == 'demo_prediction':
        cli_demo_prediction(triple_source, config, model_class)
    elif config.mode == 'profile':
        optimizer_class = utils.load_class_from_module(config.optimizer,
                                    'torch.optim')
        cli_profile(triple_source, config, model_class, optimizer_class)
    else:
        raise RuntimeError("Wrong mode {} selected.".format(config.mode))


if __name__ == '__main__':
    cli(sys.argv)
