import unittest
from kgexpr import data
from kgexpr.data import transformers
import kgekit
import kgedata
from torchvision import transforms
import numpy as np
import torch
import pytest

pytestmark = pytest.mark.random_order(disabled=True)

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
    base_seed = 5000
    enable_cuda = False


@pytest.mark.numpyfile
class DataTransformerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = Config()
        cls.triple_dir = 'kgexpr/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleDataset(cls.source.train_set, batch_size=2)
        cls.small_triple_list = next(iter(cls.dataset))
        cls.num_corrupts = 1
        cls.samples = (np.array([True, False], dtype=np.bool), cls.small_triple_list)

    def test_uniform_corruption_flag_generator(self):
        np.random.seed(0)
        corruptor = kgedata.UniformCorruptor(self.num_corrupts, 1000)
        np.testing.assert_equal(
            transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)[0],
            np.array([False, False], dtype=np.bool).reshape((-1, self.num_corrupts)))

    def test_bernoulli_corruption_generator(self):
        np.random.seed(0)
        corruptor = kgedata.BernoulliCorruptor(self.source.train_set, self.source.num_relation, self.num_corrupts, 2000)
        np.testing.assert_equal(
            transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)[0],
            np.array([False, False], dtype=np.bool).reshape((-1, self.num_corrupts)))

    def test_negative_batch_generator(self):
        np.random.seed(0)
        corruptor = kgedata.UniformCorruptor(self.num_corrupts, 1000)
        negative_sampler = kgedata.LCWANoThrowSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            1000,
            kgedata.LCWANoThrowSamplerStrategy.Hash)

        sample = transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)
        batch, negatives = transformers.NegativeBatchGenerator(negative_sampler)(sample)

        np.testing.assert_equal(
            batch, np.array([
                [0, 0, 1],
                [0, 1, 2],
            ], dtype=np.int64))
        np.testing.assert_equal(
            negatives,
            np.array([
                [2, 0, 1], [0, 1, 1],
                [2, 1, 2], [0, 0, 2],
            ],
            dtype=np.int64))

    # def test_literal_collate(self):
    #     np.random.seed(0)
    #     negative_sampler = kgedata.LCWANoThrowSampler(
    #         self.source.train_set, self.source.num_entity,
    #         self.source.num_relation, 1, 1,
    #         kgedata.LCWANoThrowSamplerStrategy.Hash)
    #     batch, negatives = collators.LiteralCollate(
    #         self.source,
    #         negative_sampler,
    #         literals=['facts'],
    #         transforms=dict(
    #             triple_transform=transformers.OrderedTripleListTransform("hrt"),
    #             fact_transform=transformers.FactTransform("hrt"),
    #         ),
    #         sample_negative_for_non_triples=False,
    #     )(self.samples, 0)
    #     np.testing.assert_equal(
    #         batch, np.array([
    #         ], dtype=np.int64))
    #     np.testing.assert_equal(
    #         negatives,
    #         np.array([
    #         ],
    #                  dtype=np.int64))
