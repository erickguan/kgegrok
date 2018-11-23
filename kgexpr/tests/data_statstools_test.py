import unittest
from kgexpr import data
from kgexpr.data import statstools
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
    epoches = 1


@pytest.mark.numpyfile
class DataStatstoolsTest(unittest.TestCase):
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

    def test_reciprocal_rank_fn(self):
        self.assertAlmostEqual(statstools.reciprocal_rank_fn(10), 0.1)

    def test_hits_reducer(self):
        reducer = statstools.HitsReducer(10)
        self.assertEqual(reducer(0, 10), 1)
        self.assertEqual(reducer(0, 9), 1)
        self.assertEqual(reducer(0, 11), 0)

    def test_calc_fns(self):
        self.assertAlmostEqual(
            statstools.calc_rank([1, 2, 4], 3), (1 + 2 + 4) / 3.0)
        self.assertAlmostEqual(
            statstools.calc_reciprocal_rank([1, 2, 4], 3),
            (1.0 + 1 / 2.0 + 1 / 4.0) / 3.0)
        self.assertAlmostEqual(statstools.calc_hits(2, [1, 2, 4], 3), 2 / 3.0)

    def test_dict_key_gen(self):
        self.assertEqual(statstools.dict_key_gen('a', 'b'), "a_b")
