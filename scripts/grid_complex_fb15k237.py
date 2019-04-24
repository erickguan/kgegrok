import logging
import numpy as np
import sys
import select
import argparse
import os.path

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from sklearn.model_selection import ParameterGrid

import sys
sys.path.append('..')
from kgegrok import data
from kgegrok import estimate
from kgegrok import stats
from kgegrok import evaluation
from kgegrok import utils
from kgegrok.stats import create_drawer

if __name__ == '__main__':
  grid = [{
      'negative_entity': [1, 5, 10, 15],
      'negative_relation': [0, 1, 5],
      'entity_embedding_dimension': [30, 50, 100, 150, 200],
      'alpha': [0.001, 0.005, 0.01, 0.1],
      'weight_decay': [0.96]
  }]

  logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S',
      level=logging.INFO)
  utils.report_gpu_info()

  default_args = {
      'data_dir': 'data/FB15k-237',
      'triple_order': 'htr',
      'delimiter': ' ',
      'batch_size': 10000,
      'num_workers': 10,
      'num_evaluation_workers': 10,
      'model': "TransE",
      'optimizer': "SGD",
      'margin': 1.0,
      'epochs': 1000,
      'lambda_': 0.001,
      'evaluation_load_factor': 0.01
  }
  config = utils.build_config_with_dict(default_args)
  print(config.__dict__)

  triple_source = data.TripleSource(config.data_dir, config.triple_order,
                                    config.triple_delimiter)
  model_class = utils.load_class_from_module(config.model, 'kgegrok.models',
                                             'kgegrok.text_models')

  evaluator = evaluation.ParallelEvaluator(config, triple_source)
  for changed_config in ParameterGrid(grid):
    d = {}
    d.update(config.__dict__)
    changed_config[
        'name'] = "TransE-FB15k237-neg_e_{}-neg_r_{}-ent_dim_{}-alpha_{}-weight_decay_{}".format(
            changed_config['negative_entity'],
            changed_config['negative_relation'],
            changed_config['entity_embedding_dimension'],
            changed_config['alpha'],
            changed_config['weight_decay'])
    d.update(changed_config)
    if os.path.exists(
        os.path.join('model_states', changed_config['name'],
                     'checkpoint.pth.tar_1000')):
      continue
    search_config = utils.build_config_with_dict(d)

    utils.seed_modules(
        config,
        numpy_seed=10000,
        torch_seed=20000,
        torcu_cuda_seed_all=2192,
        cuda_deterministic=True,
        kgegrok_base_seed=30000,
        cuda_benchmark=config.cudnn_benchmark)

    optimizer_class = utils.load_class_from_module(search_config.optimizer,
                                                   'torch.optim')
    data_loader = data.create_dataloader(triple_source, search_config,
                                         model_class.require_labels())
    drawer = stats.create_drawer(search_config)
    model = estimate.train_and_validate(
        triple_source,
        search_config,
        data_loader,
        model_class,
        optimizer_class,
        evaluator=evaluator,
        stat_gather=evaluation.build_stat_gather_from_config(config, drawer),
        drawer=drawer)
