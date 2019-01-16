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


@pytest.mark.numpyfile
class DataTransformerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = Config()
        cls.triple_dir = 'kgegrok/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleDataset(cls.source.train_set, batch_size=2)
        cls.small_triple_list = next(iter(cls.dataset))
        cls.num_corrupts = 1
        cls.samples = (np.array([True, False], dtype=np.bool), cls.small_triple_list)
        np.random.seed(0)

    def _build_corruptor(self):
        corruptor = kgedata.UniformCorruptor(self.num_corrupts, 1000)
        return corruptor

    def test_uniform_corruption_flag_generator(self):
        corruptor = self._build_corruptor()
        np.testing.assert_equal(
            transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)[0],
            np.array([False, False], dtype=np.bool).reshape((-1, self.num_corrupts)))

    def test_bernoulli_corruption_generator(self):
        corruptor = kgedata.BernoulliCorruptor(self.source.train_set, self.source.num_relation, self.num_corrupts, 2000)
        np.testing.assert_equal(
            transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)[0],
            np.array([False, False], dtype=np.bool).reshape((-1, self.num_corrupts)))

    def _build_corruptor_sampler(self):
        corruptor = self._build_corruptor()
        negative_sampler = kgedata.PerturbationSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            1000,
            kgedata.PerturbationSamplerStrategy.Hash)
        return corruptor, negative_sampler

    def _gen_sample_with_negs(self):
        corruptor, negative_sampler = self._build_corruptor_sampler()
        sample = transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)
        sample = transformers.NegativeBatchGenerator(negative_sampler)(sample)
        return sample

    def test_negative_batch_generator(self):
        batch, negatives = self._gen_sample_with_negs()
        np.testing.assert_equal(
            batch, np.array([
                [0, 0, 1],
                [0, 1, 2],
            ], dtype=np.int64))
        np.testing.assert_equal(
            negatives,
            np.array([
                [[2, 0, 1], [0, 1, 1]],
                [[2, 1, 2], [0, 0, 2]],
            ],
            dtype=np.int64))

    def test_tensor_transform(self):
        sample = self._gen_sample_with_negs()
        batch, negatives = transformers.tensor_transform(sample)
        np.testing.assert_equal(
            batch.numpy(), np.array([
                [0, 0, 1],
                [0, 1, 2],
            ], dtype=np.int64))
        np.testing.assert_equal(
            negatives.numpy(),
            np.array([
                [[2, 0, 1], [0, 1, 1]],
                [[2, 1, 2], [0, 0, 2]],
            ],
            dtype=np.int64))

    def _gen_labels(self):
        sample = self._gen_sample_with_negs()
        transform = transformers.LabelBatchGenerator(self.config)
        return transform(sample)

    def test_label_batch_generator(self):
        batch, negatives, labels = self._gen_labels()
        np.testing.assert_equal(
            labels, np.concatenate([np.ones(batch.shape[0], dtype=np.int64), np.full(negatives.shape[0]*negatives.shape[1], -1, dtype=np.int64)]))

    def test_labels_type_transform(self):
        batch, negatives, labels = self._gen_labels()
        labels = labels.astype(np.int32)
        expected_labels = np.concatenate([np.ones(batch.shape[0], dtype=np.int64), np.full(negatives.shape[0]*negatives.shape[1], -1, dtype=np.int64)])
        sample = transformers.tensor_transform([batch, negatives, labels])
        _, _, labels = transformers.labels_type_transform(sample)

        self.assertTrue(
            torch.all(torch.eq(labels, torch.from_numpy(expected_labels).float())))

    def none_none_label_batch_generator(self):
        sample = self._gen_transposed_sample_with_negs()
        _, _, labels = transformers.none_label_batch_generator(sample)
        self.assertEqual(labels, None)

    @pytest.mark.skip(reason="WIP")
    def test_literal_collate(self):
        np.random.seed(0)
        negative_sampler = kgedata.PerturbationSampler(
            self.source.train_set, self.source.num_entity,
            self.source.num_relation, 1, 1,
            kgedata.PerturbationSamplerStrategy.Hash)
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
        np.testing.assert_equal(
            batch, np.array([
            ], dtype=np.int64))
        np.testing.assert_equal(
            negatives,
            np.array([
            ],
                     dtype=np.int64))

@pytest.mark.numpyfile
class TestDataTransformerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = Config()
        cls.triple_dir = 'kgegrok/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleDataset(cls.source.test_set, batch_size=1)
        cls.small_triple_list = next(iter(cls.dataset))
        np.random.seed(0)

    def _build_triple_tile_generator(self):
        gen = transformers.TripleTileGenerator(self.config, self.source)
        return gen

    def _gen_triple_tiles(self):
        gen = self._build_triple_tile_generator()
        return gen(self.small_triple_list)

    def test_triple_tile_generator(self):
        samples, batch, splits = self._gen_triple_tiles()
        self.assertEqual(self.source.num_entity, 4)
        self.assertEqual(self.source.num_relation, 3)
        np.testing.assert_equal(batch, np.array([[1, 2, 3]], dtype=np.int64))

        np.testing.assert_equal(samples, np.array([
            [0, 2, 3], #0
            [1, 2, 3],
            [2, 2, 3],
            [3, 2, 3], #3
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3], # 7
            [1, 0, 3],
            [1, 1, 3],
            [1, 2, 3], # 10
        ], dtype=np.int64))

        self.assertEqual(splits, [(0,4,8,11)])

    def test_test_batch_transform(self):
        sample = self._gen_triple_tiles()
        tiled_t, batch, splits = transformers.test_batch_transform(sample)
        self.assertEqual(self.source.num_entity, 4)
        self.assertEqual(self.source.num_relation, 3)
        np.testing.assert_equal(batch, np.array([[1, 2, 3]], dtype=np.int64))
        np.testing.assert_equal(tiled_t[0].numpy(), np.array([
            [0, 2, 3], #0
            [1, 2, 3],
            [2, 2, 3],
            [3, 2, 3], #3
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3], # 7
            [1, 0, 3],
            [1, 1, 3],
            [1, 2, 3], # 10
        ], dtype=np.int64))

        self.assertEqual(splits, [(0,4,8,11)])



@pytest.mark.numpyfile
class DataTransformerCWASamplerTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.config = Config()
        cls.triple_dir = 'kgegrok/tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleDataset(cls.source.train_set, batch_size=2)
        cls.small_triple_list = next(iter(cls.dataset))
        cls.num_corrupts = 1
        cls.samples = (np.array([True, False], dtype=np.bool), cls.small_triple_list)
        np.random.seed(0)

    def _build_corruptor(self):
        corruptor = kgedata.UniformCorruptor(self.num_corrupts, 1000)
        return corruptor

    def _build_corruptor_sampler(self):
        corruptor = self._build_corruptor()
        negative_sampler = kgedata.CWASampler(
            self.source.num_entity,
            self.source.num_relation,
            True)
        return corruptor, negative_sampler

    def _gen_sample_with_negs(self):
        corruptor, negative_sampler = self._build_corruptor_sampler()
        sample = transformers.CorruptionFlagGenerator(corruptor)(self.small_triple_list)
        sample = transformers.NegativeBatchGenerator(negative_sampler)(sample)
        return sample

    def test_label_batch_generator(self):
        sample = self._gen_sample_with_negs()
        label_gen = kgedata.LabelGenerator(self.source.train_set)
        transform = transformers.LabelBatchGenerator(self.config, label_gen)
        batch, negatives, labels = transform(sample)
        np.testing.assert_equal(
            labels, np.array([
                1, 1,
                -1, 1, -1, -1, 1, -1, -1,
                -1, -1, 1, -1, -1, 1, -1
            ], dtype=np.int64).ravel()
        )
