import unittest
from kgegrok import data
from kgegrok.data import transformers
import kgekit
import kgedata
from torchvision import transforms
import numpy as np
import torch
import pytest
from kgegrok.stats.constants import *

pytestmark = pytest.mark.random_order(disabled=True)


@pytest.mark.numpyfile
class Config(object):
  """Mocked implementation of config"""
  data_dir = "kgegrok/tests/fixtures/triples"
  triple_order = "hrt"
  triple_delimiter = ' '
  negative_entity = 1
  negative_relation = 1
  batch_size = 2
  num_workers = 1
  entity_embedding_dimension = 10
  margin = 0.01
  epochs = 1
  base_seed = 5000
  enable_cuda = False
  report_dimension = StatisticsDimension.DEFAULT


@pytest.fixture(scope="function", autouse=True)
def seeding():
  np.random.seed(0)


@pytest.fixture(scope="module")
def num_corrupts():
  return 1


@pytest.fixture(scope="module")
def config():
  c = Config()
  return c


@pytest.fixture(scope="module")
def source():
  triple_dir = 'kgegrok/tests/fixtures/triples'
  return data.TripleSource(triple_dir, 'hrt', ' ')


@pytest.fixture(scope="module")
def small_triple_list(config, source):
  dataset = data.TripleDataset(source.train_set, batch_size=2)
  return next(iter(dataset))


@pytest.fixture(scope="module")
def corruptor(num_corrupts):
  return kgedata.UniformCorruptor(num_corrupts, 1000)


@pytest.fixture(scope="module")
def neg_sampler(source):
  return kgedata.CWASampler(source.num_entity, source.num_relation, True)


@pytest.fixture(scope="module")
def generated_sample_with_negs(corruptor, neg_sampler, small_triple_list):
  sample = transformers.CorruptionFlagGenerator(corruptor)(small_triple_list)
  sample = transformers.NegativeBatchGenerator(neg_sampler)(sample)
  return sample


def test_label_batch_generator(source, generated_sample_with_negs, config):
  label_gen = kgedata.MemoryLabelGenerator(source.train_set)
  pos_label_gen = kgedata.StaticLabelGenerator(True)
  transform = transformers.LabelBatchGenerator(config, label_gen, pos_label_gen)
  batch, negatives, labels = transform(generated_sample_with_negs)
  np.testing.assert_equal(
      labels[0],
      np.array([1, 1]))
  np.testing.assert_equal(labels[1], np.array([-1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1],
               dtype=np.int64).ravel())
