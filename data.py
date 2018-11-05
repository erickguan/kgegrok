"""Data loading and processing"""

import os
import kgekit.io
import kgekit.utils
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import random

TRIPLE_LENGTH = 3
NUM_POSITIVE_INSTANCE = 1

class TripleSource(object):
    """Triple stores."""

    def __init__(self, data_dir, triple_order, delimiter):
        """loads the data.
        Args:
            Directory with {train,valid,test}.txt
        """
        self.data_dir = data_dir
        self.trains, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "train.txt"), triple_order, delimiter)
        assert num_failed == 0
        self.valids, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "valid.txt"), triple_order, delimiter)
        assert num_failed == 0
        self.tests, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "test.txt"), triple_order, delimiter)
        assert num_failed == 0

    @property
    def train_set(self):
        return self.trains

    @property
    def valid_set(self):
        return self.valids

    @property
    def test_set(self):
        return self.tests

class TripleIndexesDataset(Dataset):
    """Loads triple indexes dataset."""

    def __init__(self, triple_source, transform=None):
        """
        Args:
            triple_source: Triple storage.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.triple_source = triple_source
        self.transform = transform

    def __len__(self):
        return len(self.triple_source.train_set)

    def __getitem__(self, idx):
        sample = self.triple_source.train_set[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class OrderedTripleTransform(object):
    """Reformat a triple index into list.

    Args:
        triple_order (str): Desired triple order in list.
    """

    def __init__(self, triple_order):
        kgekit.utils.assert_triple_order(triple_order)
        self.triple_order = triple_order

    def __call__(self, sample):
        vec = []
        for o in self.triple_order:
            if o == 'h':
                vec.append(sample.head)
            elif o == 'r':
                vec.append(sample.relation)
            elif o == 't':
                vec.append(sample.tail)

        return vec

class OrderedTripleListTransform(object):
    """Reformat a triple index into list.

    Args:
        triple_order (str): Desired triple order in list.
    """

    def __init__(self, triple_order):
        kgekit.utils.assert_triple_order(triple_order)
        self.triple_order = triple_order

    def __call__(self, samples):
        batch_size = len(samples)
        batch = np.empty((batch_size, NUM_POSITIVE_INSTANCE, TRIPLE_LENGTH), dtype=np.int32)
        for i in range(batch_size):
            for o in self.triple_order:
                t = samples[i]
                if o == 'h':
                    batch[i, 0, 0] = t.head
                elif o == 'r':
                    batch[i, 0, 1] = t.relation
                elif o == 't':
                    batch[i, 0, 2] = t.tail

        return batch

def _make_random_choice(size, probability):
    if isinstance(probability[0], float):
        choices = [np.random.choice([True, False], p=probability) for _ in range(size)]
    else:
        choices = [np.random.choice([True, False], p=probability[i]) for i in range(size)]
    return choices

def _round_probablities(probabilities):
    return [probabilities[0], 1-probabilities[0]]

class BernoulliCorruptionCollate(object):
    """Generates corrupted head/tail decision in Bernoulli Distribution based on tph.
    True means we will corrupt head."""

    def __init__(self, triple_source, bernoulli_corruptor):
        """Loads the data from source."""
        self.train_set = triple_source.train_set
        self.bernoulli_corruptor = bernoulli_corruptor

    def __call__(self, batch):
        """return corruption flag and actual batch"""
        if isinstance(batch[0], list):
            probabilities = [_round_probablities(self.bernoulli_corruptor.getProbablityRelation(t[1])) for t in batch]
        else:
            probabilities = [_round_probablities(self.bernoulli_corruptor.getProbablityRelation(t.relation)) for t in batch]
        return _make_random_choice(len(batch), probabilities), batch

class UniformCorruptionCollate(object):
    """Generates corrupted head/tail decision in uniform distribution.
    True means we will corrupt head."""

    def __call__(self, batch):
        return _make_random_choice(len(batch), [0.5, 0.5]), batch

class LCWANoThrowCollate(object):
    """Process and sample an negative batch. Supports multiprocess from pytorch.
    We don't throw if |set(subject, predict)| is empty.
    Returns:
        Positive tensor with shape (batch_size, 1, 3).
        Negative tensor with shape (batch_size, negative_samples, 3).
    """

    def __init__(self, triple_source, negative_sampler, transform=None):
        self.sampler = negative_sampler
        self.train_set = triple_source.train_set
        self.transform = transform

    def __call__(self, batch_set, sample_seed=None):
        """process a mini-batch. Each sample in the batch is a list with 3 TripleIndex"""
        if isinstance(batch_set, tuple) and len(batch_set) == 2:
            corrupt_head, batch = batch_set
        else:
            raise RuntimeError("Wrong parameter type. Need a tuple (corrupt_head, batch). Got " + str(type(batch_set)))
        batch_size = len(batch)
        arr = np.empty((batch_size, self.sampler.numNegativeSamples(), TRIPLE_LENGTH), dtype=np.int32)
        if sample_seed is not None:
            assert isinstance(sample_seed, int)
            self.sampler.sample(arr, corrupt_head, batch, sample_seed)
        else:
            self.sampler.sample(arr, corrupt_head, batch)
        if self.transform:
            batch = self.transform(batch)
        return torch.IntTensor(batch), torch.IntTensor(arr)

