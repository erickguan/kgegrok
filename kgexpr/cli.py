import logging
import numpy as np
import sys
import select
import argparse
from contextlib import contextmanager

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import visdom

from kgexpr import data
from kgexpr import estimate
from kgexpr import stats
from kgexpr import evaluation
from kgexpr import utils


def _create_drawer(config):
    return stats.ReportDrawer(visdom.Visdom(
        port=6006), config) if config.plot_graph else None

def cli_profile(triple_source, config, model_class, optimizer_class):
    with torch.autograd.profiler.profile() as prof:
        model = estimate.train(triple_source, config, model_class,
                        optimizer_class, drawer=_create_drawer(config))
    print(prof)

def cli_train(triple_source, config, model_class, optimizer_class):
    model = estimate.train(triple_source, config, model_class,
                           optimizer_class, drawer=_create_drawer(config))

def cli_train_and_validate(triple_source, config, model_class, optimizer_class, validation_process_pool):
    model = estimate.train_and_validate(triple_source, config, model_class,
                                        optimizer_class, validation_process_pool, drawer=_create_drawer(config))

def cli_test(triple_source, config, model_class, validation_process_pool):
    assert config.resume is not None and len(config.resume) > 0
    model = estimate.test(triple_source, config, model_class, validation_process_pool)


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
    parser.add_argument('--mode', default='train')
    for k in utils.Config.registered_options():
        cfg_type = utils.Config.option_type(k)
        if cfg_type == bool:
            parser.add_argument("--{}".format(k), type=utils.str2bool)
        else:
            parser.add_argument("--{}".format(k), type=utils.Config.option_type(k))

    parsed_args = vars(parser.parse_args(args[1:]))
    config = utils.Config(parsed_args)
    config.enable_cuda = True if torch.cuda.is_available(
    ) and config.enable_cuda else False

    return config, parsed_args


def seed_modules(config, numpy_seed, torch_seed, torcu_cuda_seed_all,
                 cuda_deterministic, kgexpr_base_seed, cuda_benchmark):
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torcu_cuda_seed_all)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = cuda_benchmark
    config.base_seed = kgexpr_base_seed

@contextmanager
def _validation_resource_manager(mode, config, triple_source, required_modes=['train_validate', 'test']):
    """prepare resources if validation is needed."""
    enabled = mode in required_modes
    try:
        if enabled:
            ctx = mp.get_context('spawn')
            pool = evaluation.EvaluationProcessPool(self.config, self.triple_source, self.ctx)
            pool.start()
            yield pool
        else:
            yield None
    finally:
        if enabled:
           pool.stop()

def cli(args):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    utils.report_gpu_info()

    config, parsed_args = cli_config_and_parse_args(args)
    print(config.__dict__)
    print("Continue? Starts in 10s. [Ctrl-C] to stop.")

    # let's save some seconds
    select.select([sys.stdin], [], [], 3)

    seed_modules(
        config,
        numpy_seed=10000,
        torch_seed=20000,
        torcu_cuda_seed_all=2192,
        cuda_deterministic=True,
        kgexpr_base_seed=30000,
        cuda_benchmark=config.cudnn_benchmark)

    triple_source = data.TripleSource(config.data_dir, config.triple_order,
                                      config.triple_delimiter)
    model_class = utils.load_class_from_module(config.model, 'kgexpr.models',
                                         'kgexpr.text_models')

    with _validation_resource_manager(parsed_args['mode'], config, triple_source) as pool:
        # maybe roughly 10s now
        select.select([sys.stdin], [], [], 4)

        if parsed_args['mode'] == 'train':
            optimizer_class = utils.load_class_from_module(config.optimizer,
                                                    'torch.optim')
            cli_train(triple_source, config, model_class, optimizer_class)
        elif parsed_args['mode'] == 'train_validate':
            optimizer_class = utils.load_class_from_module(config.optimizer,
                                                    'torch.optim')
            cli_train_and_validate(triple_source, config, model_class, optimizer_class, pool)
        elif parsed_args['mode'] == 'test':
            cli_test(triple_source, config, model_class, pool)
        elif parsed_args['mode'] == 'demo_prediction':
            cli_demo_prediction(triple_source, config, model_class)
        elif parsed_args['mode'] == 'profile':
            optimizer_class = utils.load_class_from_module(config.optimizer,
                                        'torch.optim')
            cli_profile(triple_source, config, model_class, optimizer_class)
        else:
            raise RuntimeError("Wrong mode {} selected.".format(parsed_args['mode']))


if __name__ == '__main__':
    cli(sys.argv)
