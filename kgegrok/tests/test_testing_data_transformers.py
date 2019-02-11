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


@pytest.fixture(scope="module")
def config():
  c = Config()
  return c


@pytest.fixture(scope="module")
def source(config):
  triple_dir = 'kgegrok/tests/fixtures/triples'
  return data.TripleSource(triple_dir, 'hrt', ' ')


@pytest.fixture(scope="module")
def small_triple_list(source):
  dataset = data.TripleDataset(source.test_set, batch_size=1)
  triple_list = next(iter(dataset))
  return triple_list


@pytest.fixture(scope="function", autouse=True)
def seeding():
  np.random.seed(0)


@pytest.fixture(scope="module")
def triple_tile_generator(config, source):
  return transformers.TripleTileGenerator(config, source)


@pytest.fixture(scope="module")
def generated_triple_tiles(triple_tile_generator, small_triple_list):
  return triple_tile_generator(small_triple_list)


def test_triple_tile_generator(source, generated_triple_tiles):
  samples, batch, splits = generated_triple_tiles
  assert source.num_entity == 4
  assert source.num_relation == 3
  np.testing.assert_equal(batch, np.array([[1, 2, 3]], dtype=np.int64))

  np.testing.assert_equal(
      samples,
      np.array(
          [
              [0, 2, 3],  #0
              [1, 2, 3],
              [2, 2, 3],
              [3, 2, 3],  #3
              [1, 2, 0],
              [1, 2, 1],
              [1, 2, 2],
              [1, 2, 3],  # 7
              [1, 0, 3],
              [1, 1, 3],
              [1, 2, 3],  # 10
          ],
          dtype=np.int64))

  assert splits[0][0] == 0
  assert splits[0][1] == 4
  assert splits[0][2] == 8
  assert splits[0][3] == 11


def test_test_batch_transform(generated_triple_tiles, source):
  tiled_t, batch, splits = transformers.test_batch_transform(
      generated_triple_tiles)
  assert source.num_entity == 4
  assert source.num_relation == 3
  np.testing.assert_equal(batch, np.array([[1, 2, 3]], dtype=np.int64))
  np.testing.assert_equal(
      tiled_t[0].numpy(),
      np.array(
          [
              [0, 2, 3],  #0
              [1, 2, 3],
              [2, 2, 3],
              [3, 2, 3],  #3
              [1, 2, 0],
              [1, 2, 1],
              [1, 2, 2],
              [1, 2, 3],  # 7
              [1, 0, 3],
              [1, 1, 3],
              [1, 2, 3],  # 10
          ],
          dtype=np.int64))

  assert splits[0][0] == 0
  assert splits[0][1] == 4
  assert splits[0][2] == 8
  assert splits[0][3] == 11
