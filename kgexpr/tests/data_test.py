import unittest
from kgexpr import data
from kgexpr.data import constants
from kgexpr.data import collators
import kgekit
from torchvision import transforms
import numpy as np
import torch
import pytest


class Config(object):
    """Mocked implementation of config"""
    data_dir = "kgexpr/tests/fixtures/triples"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 1
    negative_relation = 1
    batch_size = 100
    num_workers = 1
    entity_embedding_dimension = 10
    margin = 0.01
    epochs = 1


@pytest.mark.numpyfile
class DataTest(unittest.TestCase):

    @classmethod
    def setUp(cls):
        cls.triple_dir = 'kgexpr/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleIndexesDataset(cls.source)
        cls.small_triple_list = [
            kgekit.TripleIndex(0, 0, 1),
            kgekit.TripleIndex(1, 1, 2)
        ]
        cls.samples = ([True, False], cls.small_triple_list)

    def test_triple_source(self):
        self.assertEqual(len(self.source.train_set), 4)
        self.assertEqual(len(self.source.valid_set), 1)
        self.assertEqual(len(self.source.test_set), 1)
        self.assertEqual(self.source.num_entity, 4)
        self.assertEqual(self.source.num_relation, 3)

    def test_dataset(self):
        self.assertEqual(len(self.dataset), 4)
        self.assertEqual(self.dataset[0], kgekit.TripleIndex(0, 0, 1))

        valid_dataset = data.TripleIndexesDataset(
            self.source, constants.DatasetType.VALIDATION)
        self.assertEqual(len(valid_dataset), 1)

    def test_ordered_triple_transform(self):
        transform_dataset = data.TripleIndexesDataset(
            self.source,
            transform=transforms.Compose([data.OrderedTripleTransform("hrt")]))
        self.assertEqual(transform_dataset[0], [0, 0, 1])

    def test_get_triples_from_batch(self):
        np.random.seed(0)
        negative_sampler = kgekit.LCWANoThrowSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            kgekit.LCWANoThrowSamplerStrategy.Hash)
        batch, negatives = collators.LCWANoThrowCollate(
            self.source,
            negative_sampler,
            transform=data.OrderedTripleListTransform("hrt"))(self.samples, 0)

        batch_size = batch.shape[0]
        h, r, t = data.get_triples_from_batch(batch)
        self.assertEqual(h.shape, (2,))
        self.assertEqual(r.shape, (2,))
        self.assertEqual(t.shape, (2,))
        np.testing.assert_equal(h, np.array([0, 1], dtype=np.int64))
        np.testing.assert_equal(r, np.array([0, 1], dtype=np.int64))
        np.testing.assert_equal(t, np.array([1, 2], dtype=np.int64))

    def test_get_negative_samples_from_batch(self):
        np.random.seed(0)
        negative_sampler = kgekit.LCWANoThrowSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            kgekit.LCWANoThrowSamplerStrategy.Hash)
        batch, negatives = collators.LCWANoThrowCollate(
            self.source,
            negative_sampler,
            transform=data.OrderedTripleListTransform("hrt"))(self.samples, 0)

        h, r, t = data.get_triples_from_batch(negatives)
        self.assertEqual(h.shape, (2, 2))
        self.assertEqual(r.shape, (2, 2))
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_equal(h, np.array([
            [0, 0],
            [0, 1],
        ], dtype=np.int64))
        np.testing.assert_equal(r, np.array([
            [0, 2],
            [1, 0],
        ], dtype=np.int64))
        np.testing.assert_equal(t, np.array([
            [3, 1],
            [2, 2],
        ], dtype=np.int64))

    def test_get_labels_from_batch(self):
        np.random.seed(0)
        negative_sampler = kgekit.LCWANoThrowSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            kgekit.LCWANoThrowSamplerStrategy.Hash)
        batch_sampled = collators.LCWANoThrowCollate(
            self.source,
            negative_sampler,
            transform=data.OrderedTripleListTransform("hrt"))(self.samples, 0)
        batch, negatives = collators.label_collate(batch_sampled)

        h, r, t, y = data.get_triples_from_batch(batch)
        np.testing.assert_equal(h, np.array([0, 1], dtype=np.int64))
        np.testing.assert_equal(r, np.array([0, 1], dtype=np.int64))
        np.testing.assert_equal(t, np.array([1, 2], dtype=np.int64))
        np.testing.assert_equal(y, np.array([1, 1], dtype=np.int64))

        h, r, t, y = data.get_triples_from_batch(negatives)
        np.testing.assert_equal(h, np.array([
            [0, 0],
            [0, 1],
        ], dtype=np.int64))
        np.testing.assert_equal(r, np.array([
            [0, 2],
            [1, 0],
        ], dtype=np.int64))
        np.testing.assert_equal(t, np.array([
            [3, 1],
            [2, 2],
        ], dtype=np.int64))
        np.testing.assert_equal(y,
                                np.array([
                                    [-1, -1],
                                    [-1, -1],
                                ], dtype=np.int64))

    def test_data_loader(self):
        np.random.seed(0)
        config = Config()
        data_loader = data.create_dataloader(self.source, config, False,
                                             constants.DatasetType.TRAINING)
        _, sample_batched = next(enumerate(data_loader))
        batch, negatives = sample_batched
        np.testing.assert_equal(
            batch,
            np.array([[[0, 0, 1]], [[0, 1, 2]], [[1, 2, 3]], [[3, 1, 2]]],
                     dtype=np.int64))
        np.testing.assert_equal(
            negatives[0, :, :], np.array([
                [0, 0, 1],
                [0, 1, 1],
            ],
                                         dtype=np.int64))

    def test_data_loader_with_label(self):
        np.random.seed(0)
        config = Config()
        data_loader = data.create_dataloader(self.source, config, True,
                                             constants.DatasetType.TRAINING)
        _, sample_batched = next(enumerate(data_loader))
        batch, negatives = sample_batched
        np.testing.assert_equal(
            batch,
            np.array([[[0, 0, 1, 1]], [[0, 1, 2, 1]], [[1, 2, 3, 1]],
                      [[3, 1, 2, 1]]],
                     dtype=np.int64))
        self.assertEqual(negatives.shape, (4, 2, 4))
        np.testing.assert_equal(
            negatives[0, :, :],
            np.array([
                [0, 0, 1, -1],
                [0, 1, 1, -1],
            ], dtype=np.int64))

    def test_expand_triple_to_sets(self):
        h, r, t = data.expand_triple_to_sets((1, 2, 3), 10,
                                             constants.TripleElement.HEAD)
        np.testing.assert_equal(h, np.arange(10, dtype=np.int64))
        np.testing.assert_equal(r, np.tile(np.array([2], dtype=np.int64), 10))
        np.testing.assert_equal(t, np.tile(np.array([3], dtype=np.int64), 10))

        h, r, t = data.expand_triple_to_sets((1, 2, 3), 10,
                                             constants.TripleElement.RELATION)
        np.testing.assert_equal(h, np.tile(np.array([1], dtype=np.int64), 10))
        np.testing.assert_equal(r, np.arange(10, dtype=np.int64))
        np.testing.assert_equal(t, np.tile(np.array([3], dtype=np.int64), 10))

        h, r, t = data.expand_triple_to_sets((1, 2, 3), 10,
                                             constants.TripleElement.TAIL)
        np.testing.assert_equal(h, np.tile(np.array([1], dtype=np.int64), 10))
        np.testing.assert_equal(r, np.tile(np.array([2], dtype=np.int64), 10))
        np.testing.assert_equal(t, np.arange(10, dtype=np.int64))

        with pytest.raises(RuntimeError):
            h, r, t = data.expand_triple_to_sets((1, 2, 3), 10, -1)
