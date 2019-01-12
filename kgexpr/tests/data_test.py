import unittest
from kgexpr import data
from kgexpr.data import constants
from kgexpr.data import transformers
from kgexpr.stats.constants import StatisticsDimension
import kgekit
from torchvision import transforms
import numpy as np
import torch
import pytest
import kgedata

pytestmark = pytest.mark.random_order(disabled=True)

class Config(object):
    """Mocked implementation of config"""
    data_dir = "kgexpr/tests/fixtures/triples"
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


@pytest.mark.numpyfile
class DataTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.triple_dir = 'kgexpr/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.small_triple_list = [
            kgedata.TripleIndex(0, 0, 1),
            kgedata.TripleIndex(1, 1, 2)
        ]
        cls.samples = (np.array([True, False], dtype=np.bool), cls.small_triple_list)
        np.random.seed(0)

    def test_triple_source(self):
        self.assertEqual(len(self.source.train_set), 5)
        self.assertEqual(len(self.source.valid_set), 1)
        self.assertEqual(len(self.source.test_set), 1)
        self.assertEqual(self.source.num_entity, 4)
        self.assertEqual(self.source.num_relation, 3)
        np.testing.assert_equal(self.source.test_set, np.array([[1, 2, 3]], dtype=np.int64))
    
    def test_SequentialBatchSampler(self):
        dataset = data.TripleDataset(self.source.test_set)
        sampler = data.SequentialBatchSampler(dataset)
        it1 = iter(sampler)
        next(it1)
        with pytest.raises(StopIteration):
            next(it1)
        it2 = iter(sampler)
        next(it2)
        with pytest.raises(StopIteration):
            next(it2)

    def test_dataset_tensor(self):
        dataset = data.TripleDataset(self.source.train_set, batch_size=2)
        self.assertEqual(len(dataset), 3)
        np.testing.assert_equal(dataset.data[0], np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64))
        np.testing.assert_equal(dataset.data[-1], np.array([[2, 0, 3]], dtype=np.int64))

    def test_data_loader(self):
        config = Config()
        data_loader = data.create_dataloader(self.source, config, False,
                                             constants.DatasetType.TRAINING)
        sample_batched = next(iter(data_loader))
        batch, negatives, labels = sample_batched
        self.assertEqual(batch.shape, (5, 3))
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

    def test_data_loader_with_label(self):
        config = Config()
        data_loader = data.create_dataloader(self.source, config, True,
                                             constants.DatasetType.TRAINING)
        sample_batched = next(iter(data_loader))
        batch, negatives, labels = sample_batched
        self.assertEqual(batch.shape, (5, 3))
        np.testing.assert_equal(
            batch.numpy(),
            np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3],
                      [3, 1, 2], [2, 0, 3]],
                     dtype=np.int64))
        np.testing.assert_equal(batch.transpose(0, 1)[0].numpy(), np.array([0, 0, 1, 3, 2], dtype=np.int64))
        self.assertEqual(negatives.shape, (5, 2, 3))
        np.testing.assert_equal(
            negatives.numpy()[0, :],
            np.array([
                [0, 0, 0],
                [0, 2, 1],
            ], dtype=np.int64))
        labels = labels.numpy()
        self.assertEqual(labels.shape, (15,))
        np.testing.assert_equal(
            labels,
            np.array([
                1, 1, 1, 1, 1,
                -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1
                ]))

    def test_validation_dataloader(self):
        config = Config()
        data_loader = data.create_dataloader(self.source, config, False,
                                             constants.DatasetType.VALIDATION)
        batch, original, splits = next(iter(data_loader))
        batch, negatives, labels = batch
        self.assertEqual(batch.shape, (3, 11))
        np.testing.assert_equal(
            batch.numpy().T,
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
        print(negatives)
        self.assertEqual(negatives, None)
        self.assertEqual(labels, None)