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
  batch_worker_timeout = 2
  num_workers = 0
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
  np.testing.assert_equal(labels[0], np.array([1, 1]))
  np.testing.assert_equal(
      labels[1],
      np.array([-1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1],
               dtype=np.int64).ravel())


import kgegrok.utils


@pytest.mark.usefixture("monkeypatch")
@pytest.fixture(scope="function")
def cuda_device_2(monkeypatch):
  monkeypatch.setattr(kgegrok.utils, "num_cuda_devices", lambda: 2)
  assert kgegrok.utils.num_cuda_devices() == 2


def test_cwa_training_dataloader(config, source, cuda_device_2):
  config.enable_cuda = True
  assert kgegrok.utils.num_cuda_devices() == 2

  data_loader = data.create_cwa_training_dataloader(source, config)
  sample_batched = next(iter(data_loader))
  batch, negatives, labels = sample_batched
  assert batch is None
  assert negatives.shape == (2, 4, 3)
  np.testing.assert_equal(
      negatives.numpy()[0, :],
      np.array([
          [0, 0, 0],
          [0, 0, 1],
          [0, 0, 2],
          [0, 0, 3],
      ], dtype=np.int64))
  pos_labels, neg_labels = labels
  assert pos_labels is None
  neg_labels = neg_labels.numpy()
  assert neg_labels.shape == (8,)
  np.testing.assert_equal(neg_labels, np.array([-1, 1, -1, -1, -1, -1, 1, -1]))
