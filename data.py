"""Data loading and processing"""

import os
import kgekit.io
import kgekit.utils
import torch
from torch.autograd.variable import Variable
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
import random
from enum import Enum, Flag, auto
from torchvision import transforms
import functools

TRIPLE_LENGTH = 3
NUM_POSITIVE_INSTANCE = 1

DatasetType = Enum('DatasetType', 'TRAINING VALIDATION TESTING')

class LinkPredictionStatistics(Flag):
    MEAN_RECIPROCAL_RANK = auto()
    MEAN_RANK = auto()
    ALL_RANK = MEAN_RECIPROCAL_RANK | MEAN_RANK
    HITS_1 = auto()
    HITS_3 = auto()
    HITS_5 = auto()
    HITS_10 = auto()
    MINIMAL_HIT = HITS_1 | HITS_10
    ALL_HIT = HITS_1 | HITS_3 | HITS_5 | HITS_10
    DEFAULT = MINIMAL_HIT | MEAN_RECIPROCAL_RANK
    ALL = ALL_RANK | ALL_HIT

class TripleElement(Flag):
    HEAD = 1
    RELATION = 2
    TAIL = 3

    @classmethod
    def has_value(cls, value):
        return any(value == item for item in cls)

class TripleSource(object):
    """Triple stores."""

    TRAIN_FILENAME = "train.txt"
    VALID_FILENAME = "valid.txt"
    TEST_FILENAME = "test.txt"

    def __init__(self, data_dir, triple_order, delimiter):
        """loads the data.
        Args:
            Directory with {train,valid,test}.txt
        """
        self.data_dir = data_dir
        self._train_set, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, self.TRAIN_FILENAME), triple_order, delimiter)
        assert num_failed == 0
        self._valid_set, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, self.VALID_FILENAME), triple_order, delimiter)
        assert num_failed == 0
        self._test_set, num_failed = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, self.TEST_FILENAME), triple_order, delimiter)
        assert num_failed == 0
        head_compare = lambda x: x.head
        tail_compare = lambda x: x.tail
        relation_compare = lambda x: x.relation
        max_head = max([max(triple_set, key=head_compare) for triple_set in [self._train_set, self._valid_set, self._test_set]], key=head_compare)
        max_tail = max([max(triple_set, key=tail_compare) for triple_set in [self._train_set, self._valid_set, self._test_set]], key=tail_compare)
        self._num_entity = max(max_head.head, max_tail.tail) + 1
        self._num_relation = max([max(triple_set, key=relation_compare) for triple_set in [self._train_set, self._valid_set, self._test_set]], key=relation_compare).relation + 1

    @property
    def train_set(self):
        return self._train_set

    @property
    def valid_set(self):
        return self._valid_set

    @property
    def test_set(self):
        return self._test_set

    @property
    def num_entity(self):
        return self._num_entity

    @property
    def num_relation(self):
        return self._num_relation

class TripleIndexesDataset(Dataset):
    """Loads triple indexes dataset."""

    def __init__(self, triple_source, dataset_type=DatasetType.TRAINING, transform=None):
        """
        Args:
            triple_source: Triple storage.
            dataset_type: Choose the type of dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if dataset_type == DatasetType.TRAINING:
            self.triples = triple_source.train_set
        elif dataset_type == DatasetType.VALIDATION:
            self.triples = triple_source.valid_set
        elif dataset_type == DatasetType.TESTING:
            self.triples = triple_source.test_set
        else:
            raise RuntimeError("DatasetType doesn't exists. It's " + str(dataset_type))
        self.transform = transform

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        sample = self.triples[idx]

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
        batch = np.empty((batch_size, NUM_POSITIVE_INSTANCE, TRIPLE_LENGTH), dtype=np.int64)
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

class NumpyCollate(object):
    """Process triples and put them into a triple index.
    Returns:
        Positive tensor with shape (batch_size, 1, 3).
    """

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, batch):
        """process a mini-batch."""
        batch_size = len(batch)
        if self.transform:
            batch = self.transform(batch)
        batch = np.array(batch, dtype=np.int64)[:, np.newaxis, :]
        return batch

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
        if isinstance(batch_set, tuple) and len(batch_set) == 2:
            corrupt_head, batch = batch_set
        else:
            raise RuntimeError("Wrong parameter type. Need a tuple (corrupt_head, batch). Got " + str(type(batch_set)))
        batch_size = len(batch)
        arr = np.empty((batch_size, self.sampler.numNegativeSamples(), TRIPLE_LENGTH), dtype=np.int64)
        if sample_seed is not None:
            assert isinstance(sample_seed, int)
            self.sampler.sample(arr, corrupt_head, batch, sample_seed)
        else:
            self.sampler.sample(arr, corrupt_head, batch)
        if self.transform:
            batch = self.transform(batch)
        return batch, arr


def get_triples_from_batch(batch):
    """Returns h, r, t from batch."""

    NUM_TRIPLE_ELEMENT = 3

    batch_size = len(batch) if isinstance(batch, list) else batch.shape[0]
    h, r, t = np.split(batch, NUM_TRIPLE_ELEMENT, axis=2)
    return h.reshape(batch_size), r.reshape(batch_size), t.reshape(batch_size)

def get_negative_samples_from_batch(batch):
    """Returns h, r, t from batch."""

    NUM_TRIPLE_ELEMENT = 3

    batch_size, num_negative_samples, _ = batch.shape
    h, r, t = np.split(batch, NUM_TRIPLE_ELEMENT, axis=2)
    return h.reshape(batch_size, num_negative_samples), r.reshape(batch_size, num_negative_samples), t.reshape(batch_size, num_negative_samples)

def get_all_instances_from_batch(batch, negative_batch):
    """Returns h, r, t from batch and negatives."""

    pos_h, pos_r, pos_t = get_triples_from_batch(batch)
    neg_h, neg_r, neg_t = get_negative_samples_from_batch(negative_batch)

    return np.vstack((pos_h, neg_h)), np.vstack((pos_r, neg_r)), np.vstack((pos_t, neg_t))

def convert_triple_tuple_to_torch(batch):
    h, r, t = batch
    if torch.cuda.is_available():
        return (Variable(torch.from_numpy(h)).cuda(),
            Variable(torch.from_numpy(r)).cuda(),
            Variable(torch.from_numpy(t)).cuda())
    else:
        return (Variable(torch.from_numpy(h)),
            Variable(torch.from_numpy(r)),
            Variable(torch.from_numpy(t)))

def expand_triple_to_sets(triple, num_expands, arange_target):
    """Tiles triple into a large sets for testing. One node will be initialized with arange.
    Returns (h, r, t), each with a shape of (num_expands,)
    """

    if not TripleElement.has_value(arange_target):
        raise RuntimeError("arange_target is set to wrong value. It has to be one of TripleElement but it's " + str(arange_target))
    h, r, t = triple

    if arange_target == TripleElement.HEAD:
        h = np.arange(num_expands, dtype=np.int64)
        r = np.tile(np.array([r], dtype=np.int64), num_expands)
        t = np.tile(np.array([t], dtype=np.int64), num_expands)
    elif arange_target == TripleElement.RELATION:
        h = np.tile(np.array([h], dtype=np.int64), num_expands)
        r = np.arange(num_expands, dtype=np.int64)
        t = np.tile(np.array([t], dtype=np.int64), num_expands)
    elif arange_target == TripleElement.TAIL:
        h = np.tile(np.array([h], dtype=np.int64), num_expands)
        r = np.tile(np.array([r], dtype=np.int64), num_expands)
        t = np.arange(num_expands, dtype=np.int64)
    else:
        raise RuntimeError("Miracle happened. arange_target passed the validation and reached impossible branch.")

    return (h, r, t)

def create_dataloader(triple_source, config, dataset_type=DatasetType.TRAINING):
    """Creates dataloader with certain types"""
    # Use those C++ extension is fast but then we can't use spawn method to start data loader.
    if dataset_type == DatasetType.TRAINING:
        dataset = TripleIndexesDataset(triple_source, dataset_type)
        negative_sampler = kgekit.LCWANoThrowSampler(
            triple_source.train_set,
            triple_source.num_entity,
            triple_source.num_relation,
            config.negative_entity,
            config.negative_relation,
            kgekit.LCWANoThrowSamplerStrategy.Hash
        )
        corruptor = kgekit.BernoulliCorruptor(triple_source.train_set)

        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True, # May cause system froze because of non-preemption
            collate_fn=transforms.Compose([
                BernoulliCorruptionCollate(triple_source, corruptor),
                LCWANoThrowCollate(triple_source, negative_sampler, transform=OrderedTripleListTransform(config.triple_order)),
            ])
        )
    else:
        dataset = TripleIndexesDataset(triple_source, dataset_type, transform=OrderedTripleTransform(config.triple_order))
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True, # May cause system froze because of of non-preemption
            collate_fn=NumpyCollate(),
        )
    return data_loader

def reciprocal_rank_fn(rank):
    return 1.0/rank

class HitsReducer(object):
    """Used with accumulation function"""

    def __init__(self, target):
        self.target = target

    def __call__(self, rank):
        return 1 if rank <= self.target else 0

def get_rank_statistics(rank_list, features):
    result = {}
    if LinkPredictionStatistics.MEAN_RECIPROCAL_RANK & features:
        result['mean_reciprocal_rank'] = sum(map(reciprocal_rank_fn, rank_list)) / len(rank_list)
    if LinkPredictionStatistics.MEAN_RANK & features:
        result['mean_rank'] = sum(rank_list) / len(rank_list)
    if LinkPredictionStatistics.HITS_1 & features:
        result['HITS_1'] = functools.reduce(HitsReducer(1), rank_list)
    if LinkPredictionStatistics.HITS_3 & features:
        result['HITS_3'] = functools.reduce(HitsReducer(3), rank_list)
    if LinkPredictionStatistics.HITS_5 & features:
        result['HITS_5'] = functools.reduce(HitsReducer(5), rank_list)
    if LinkPredictionStatistics.HITS_10 & features:
        result['HITS_10'] = functools.reduce(HitsReducer(10), rank_list)
    return result


