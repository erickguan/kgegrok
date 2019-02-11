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
def config():
  c = Config()
  return c


@pytest.fixture(scope="module")
def num_corrupts():
  return 1


@pytest.fixture(scope="module")
def source(config):
  triple_dir = 'kgegrok/tests/fixtures/triples'
  return data.TripleSource(triple_dir, 'hrt', ' ')


@pytest.fixture(scope="module")
def small_triple_list(source):
  dataset = data.TripleDataset(source.train_set, batch_size=2)
  triple_list = next(iter(dataset))
  return triple_list


@pytest.fixture(scope="module")
def corruptor(num_corrupts):
  corruptor = kgedata.UniformCorruptor(num_corrupts, 1000)
  return corruptor


def test_uniform_corruption_flag_generator(corruptor, num_corrupts,
                                           small_triple_list):
  np.testing.assert_equal(
      transformers.CorruptionFlagGenerator(corruptor)(small_triple_list)[0],
      np.array([False, False], dtype=np.bool).reshape((-1, num_corrupts)))


def test_bernoulli_corruption_generator(source, num_corrupts,
                                        small_triple_list):
  corruptor = kgedata.BernoulliCorruptor(source.train_set, source.num_relation,
                                         num_corrupts, 2000)
  np.testing.assert_equal(
      transformers.CorruptionFlagGenerator(corruptor)(small_triple_list)[0],
      np.array([False, False], dtype=np.bool).reshape((-1, num_corrupts)))


@pytest.fixture(scope="module")
def negative_sampler(source):
  negative_sampler = kgedata.PerturbationSampler(
      source.train_set, source.num_entity, source.num_relation, 1, 1, 1000,
      kgedata.PerturbationSamplerStrategy.Hash)
  return negative_sampler


@pytest.fixture(scope="module")
def negative_sample_with_negs(small_triple_list, corruptor, negative_sampler):
  sample = transformers.CorruptionFlagGenerator(corruptor)(small_triple_list)
  sample = transformers.NegativeBatchGenerator(negative_sampler)(sample)
  return sample


def test_negative_batch_generator(negative_sample_with_negs):
  batch, negatives = negative_sample_with_negs
  np.testing.assert_equal(batch,
                          np.array([
                              [0, 0, 1],
                              [0, 1, 2],
                          ], dtype=np.int64))
  np.testing.assert_equal(
      negatives,
      np.array([
          [[0, 0, 2], [0, 1, 1]],
          [[0, 1, 3], [0, 0, 2]],
      ],
               dtype=np.int64))


def test_tensor_transform(negative_sample_with_negs):
  batch, negatives = transformers.tensor_transform(negative_sample_with_negs)
  np.testing.assert_equal(batch.numpy(),
                          np.array([
                              [0, 0, 1],
                              [0, 1, 2],
                          ], dtype=np.int64))
  np.testing.assert_equal(
      negatives.numpy(),
      np.array([
          [[0, 0, 2], [0, 1, 1]],
          [[0, 1, 3], [0, 0, 2]],
      ],
               dtype=np.int64))


@pytest.fixture(scope="module")
def generated_labels(config, negative_sample_with_negs):
  transform = transformers.LabelBatchGenerator(config)
  return transform(negative_sample_with_negs)


def test_label_batch_generator(generated_labels):
  batch, negatives, labels = generated_labels
  np.testing.assert_equal(
      labels,
      np.concatenate([
          np.ones(batch.shape[0], dtype=np.int64),
          np.full(negatives.shape[0] * negatives.shape[1], -1, dtype=np.int64)
      ]))


def test_labels_type_transform(generated_labels):
  batch, negatives, labels = generated_labels
  labels = labels.astype(np.int32)
  expected_labels = np.concatenate([
      np.ones(batch.shape[0], dtype=np.int64),
      np.full(negatives.shape[0] * negatives.shape[1], -1, dtype=np.int64)
  ])
  sample = transformers.tensor_transform([batch, negatives, labels])
  _, _, labels = transformers.labels_type_transform(sample)

  assert True == torch.all(
      torch.eq(labels,
               torch.from_numpy(expected_labels).float()))


def none_none_label_batch_generator(negative_sample_with_negs):
  _, _, labels = transformers.none_label_batch_generator(
      negative_sample_with_negs)
  assert labels == None


@pytest.mark.skip(reason="WIP")
def test_literal_collate():
  np.random.seed(0)
  negative_sampler = kgedata.PerturbationSampler(
      self.source.train_set, self.source.num_entity, self.source.num_relation,
      1, 1, kgedata.PerturbationSamplerStrategy.Hash)
  batch, negatives = collators.LiteralCollate(
      self.source,
      negative_sampler,
      literals=['facts'],
      transforms=dict(
          triple_transform=transformers.OrderedTripleListTransform("hrt"),
          fact_transform=transformers.FactTransform("hrt"),
      ),
      sample_negative_for_non_triples=False,
  )(self.samples, 0)
  np.testing.assert_equal(batch, np.array([], dtype=np.int64))
  np.testing.assert_equal(negatives, np.array([], dtype=np.int64))
