import unittest
from kgegrok import data
from kgegrok.data import constants
from kgegrok.data import transformers
from kgegrok.stats.constants import StatisticsDimension
import kgekit
from torchvision import transforms
import numpy as np
import torch
import pytest
import kgedata

pytestmark = pytest.mark.random_order(disabled=True)

class Config(object):
    """Mocked implementation of config"""
    data_dir = "kgegrok/tests/fixtures/triples"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 1
    negative_relation = 1
    batch_size = 100
    batch_worker_timeout=0
    evaluation_load_factor=0.1
    num_workers = 1
    entity_embedding_dimension = 10
    margin = 0.01
    epochs = 1
    base_seed = 1000
    enable_cuda = False
    report_dimension = StatisticsDimension.DEFAULT

@pytest.fixture(scope="module")
def config():
    return Config()

@pytest.fixture(scope="module")
def source():
    triple_dir = 'kgegrok/tests/fixtures/triples'
    return data.TripleSource(triple_dir, 'hrt', ' ')

def test_triple_source(source):
    assert len(source.train_set) == 5
    assert len(source.valid_set) == 1
    assert len(source.test_set) == 1
    assert source.num_entity == 4
    assert source.num_relation == 3
    np.testing.assert_equal(source.test_set, np.array([[1, 2, 3]], dtype=np.int64))

def test_SequentialBatchSampler(config, source):
    dataset = data.TripleDataset(source.test_set)
    sampler = data.SequentialBatchSampler(dataset)
    it1 = iter(sampler)
    next(it1)
    with pytest.raises(StopIteration):
        next(it1)
    it2 = iter(sampler)
    next(it2)
    with pytest.raises(StopIteration):
        next(it2)

def test_dataset_tensor(source, config):
    dataset = data.TripleDataset(source.train_set, batch_size=2)
    assert len(dataset) == 3
    np.testing.assert_equal(dataset._data[0], np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64))
    np.testing.assert_equal(dataset._data[-1], np.array([[2, 0, 3]], dtype=np.int64))

def test_data_loader(source, config):
    config = Config()
    data_loader = data.create_dataloader(source, config, False, constants.DatasetType.TRAINING)
    sample_batched = next(iter(data_loader))
    batch, negatives, labels = sample_batched
    assert batch.shape == (5, 3)
    np.testing.assert_equal(
        batch.numpy(),
        np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3],
                    [3, 1, 2], [2, 0, 3]],
                    dtype=np.int64))
    np.testing.assert_equal(
        negatives.numpy()[0, :], np.array([
            [0, 0, 0],
            [0, 2, 1],
        ], dtype=np.int64))

def test_data_loader_with_label(source, config):
    config = Config()
    data_loader = data.create_dataloader(source, config, True, constants.DatasetType.TRAINING)
    sample_batched = next(iter(data_loader))
    batch, negatives, labels = sample_batched
    assert batch.shape == (5, 3)
    np.testing.assert_equal(
        batch.numpy(),
        np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3],
                    [3, 1, 2], [2, 0, 3]],
                    dtype=np.int64))
    np.testing.assert_equal(batch.transpose(0, 1)[0].numpy(), np.array([0, 0, 1, 3, 2], dtype=np.int64))
    assert negatives.shape == (5, 2, 3)
    np.testing.assert_equal(
        negatives.numpy()[0, :],
        np.array([
            [0, 0, 0],
            [0, 2, 1],
        ], dtype=np.int64))
    labels = labels.numpy()
    assert labels.shape == (15,)
    np.testing.assert_equal(
        labels,
        np.array([
            1, 1, 1, 1, 1,
            -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1
            ]))

def test_validation_dataloader(source, config):
    config = Config()
    data_loader = data.create_dataloader(source, config, False, constants.DatasetType.VALIDATION)
    batch, original, splits = next(iter(data_loader))
    batch, negatives, labels = batch
    assert batch.shape == (11, 3)
    np.testing.assert_equal(
        batch.numpy(),
        np.array([
            [0, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [3, 1, 2],
            [3, 1, 0],
            [3, 1, 1],
            [3, 1, 2],
            [3, 1, 3],
            [3, 0, 2],
            [3, 1, 2],
            [3, 2, 2],
            ], dtype=np.int64))
    assert negatives == None
    assert labels == None

import kgegrok.utils

@pytest.mark.usefixture("monkeypatch")
@pytest.fixture(scope="function")
def cuda_device_2(monkeypatch):
    monkeypatch.setattr(kgegrok.utils, "num_cuda_devices", lambda: 2)
    assert kgegrok.utils.num_cuda_devices() == 2

def test_dataloader_padding(config, source, cuda_device_2):
    config.enable_cuda = True
    assert kgegrok.utils.num_cuda_devices() == 2

    data_loader = data.create_dataloader(source, config, True, constants.DatasetType.TRAINING)
    sample_batched = next(iter(data_loader))
    batch, negatives, labels = sample_batched
    assert batch.shape == (6, 3)
    np.testing.assert_equal(
        batch.numpy(),
        np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3],
                    [3, 1, 2], [2, 0, 3], [0, 0, 1]], dtype=np.int64))
    np.testing.assert_equal(batch.transpose(0, 1)[0].numpy(), np.array([0, 0, 1, 3, 2, 0], dtype=np.int64))
    assert negatives.shape == (6, 2, 3)
    np.testing.assert_equal(
        negatives.numpy()[0, :],
        np.array([
            [0, 0, 0],
            [0, 2, 1],
        ], dtype=np.int64))
    labels = labels.numpy()
    assert labels.shape == (18,)
    np.testing.assert_equal(
        labels,
        np.array([
            1, 1, 1, 1, 1, 1,
            -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1
            ]))
