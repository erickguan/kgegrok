import unittest
import data
import kgekit
from torchvision import transforms
import numpy as np
import torch
import pytest

@pytest.mark.numpyfile
class DataTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.triple_dir = 'tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleIndexesDataset(cls.source)
        cls.small_triple_list = [kgekit.TripleIndex(0, 0, 1), kgekit.TripleIndex(1, 1, 2)]
        cls.samples = ([True, False], cls.small_triple_list)

    def test_triple_source(self):
        self.assertEqual(len(self.source.train_set), 4)
        self.assertEqual(len(self.source.valid_set), 1)
        self.assertEqual(len(self.source.test_set), 1)

    def test_dataset(self):
        self.assertEqual(len(self.dataset), 4)
        self.assertEqual(self.dataset[0], kgekit.TripleIndex(0, 0, 1))

    def test_ordered_triple_transform(self):
        transform_dataset = data.TripleIndexesDataset(self.source, transform=transforms.Compose([
            data.OrderedTripleTransform("hrt")
        ]))
        self.assertEqual(transform_dataset[0], [0, 0, 1])

    def test_uniform_collate(self):
        np.random.seed(0)
        self.assertEqual(data.UniformCorruptionCollate()(self.small_triple_list), ([False, False], self.small_triple_list))

    def test_bernoulli_corruption_collate(self):
        np.random.seed(0)
        corruptor = kgekit.BernoulliCorruptor(self.source.train_set)
        self.assertEqual(data.BernoulliCorruptionCollate(self.source, corruptor)(self.small_triple_list),
            ([False, False], self.small_triple_list))

    def test_lcwa_no_throw_collate(self):
        np.random.seed(0)
        negative_sampler = kgekit.LCWANoThrowSampler(self.source.train_set, 1, 1, kgekit.LCWANoThrowSamplerStrategy.Hash)
        batch, negatives = data.LCWANoThrowCollate(self.source, negative_sampler, transform=data.OrderedTripleListTransform("hrt"))(self.samples, 0)
        np.testing.assert_equal(batch.numpy(), np.array([
            [[0, 0, 1]],
            [[1, 1, 2]],
        ], dtype=np.int32))
        np.testing.assert_equal(negatives.numpy(), np.array([
            [[0, 0, 3], [0, 2, 1]],
            [[0, 1, 2], [1, 0, 2]],
        ], dtype=np.int32))
