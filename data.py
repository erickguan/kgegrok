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
    MEAN_FILTERED_RECIPROCAL_RANK = auto()
    MEAN_RANK = auto()
    MEAN_FILTERED_RANK = auto()
    MEAN_RANKS = MEAN_RECIPROCAL_RANK | MEAN_RANK
    MEAN_FILTERED_RANKS = MEAN_FILTERED_RECIPROCAL_RANK | MEAN_FILTERED_RANK
    ALL_RANKS = MEAN_RANKS | MEAN_FILTERED_RANKS
    HITS_1 = auto()
    HITS_1_FILTERED = auto()
    HITS_3 = auto()
    HITS_3_FILTERED = auto()
    HITS_5 = auto()
    HITS_5_FILTERED = auto()
    HITS_10 = auto()
    HITS_10_FILTERED = auto()
    MINIMAL_HITS = HITS_1 | HITS_10
    MINIMAL_HITS_FILTERED = HITS_1_FILTERED | HITS_10_FILTERED
    HITS = HITS_1 | HITS_3 | HITS_5 | HITS_10
    HITS_FILTERED = HITS_1_FILTERED | HITS_3_FILTERED | HITS_5_FILTERED | HITS_10_FILTERED
    ALL_HITS = HITS | HITS_FILTERED
    UNFILTERED_DEFAULT = MINIMAL_HITS | MEAN_RECIPROCAL_RANK
    DEFAULT = MINIMAL_HITS_FILTERED | MEAN_FILTERED_RECIPROCAL_RANK
    ALL = ALL_RANKS | ALL_HITS

class StatisticsDimension(Flag):
    SEPERATE_ENTITY = auto()
    COMBINED_ENTITY = auto()
    RELATION = auto()
    DEFAULT = COMBINED_ENTITY | RELATION
    ALL = SEPERATE_ENTITY | RELATION

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

def _round_probablities(probability):
    return (probability, 1-probability)

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


class TripleTileCollate(object):
    """Process triples and put them into a tiled but flat numpy array.
    Args:
        config: config object for reading dimension information
        triple_source: triple source function
    Returns:
        Positive tensor with shape (batch_size * varied_size, 1, 3).
        varied_size will depends on testing dimension, num_entity and num_relation.
    """

    def __init__(self, config, triple_source):
        self.config = config
        self.triple_source = triple_source

    def __call__(self, batch):
        """process a mini-batch."""
        sampled, splits = kgekit.expand_triple_batch(batch,
            self.triple_source.num_entity,
            self.triple_source.num_relation,
            (self.config.report_dimension & StatisticsDimension.SEPERATE_ENTITY) or (self.config.report_dimension & StatisticsDimension.COMBINED_ENTITY),
            self.config.report_dimension & StatisticsDimension.RELATION)

        sampled = sampled[:, np.newaxis, :]
        return (sampled, batch, splits)


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
        if sample_seed is None:
            sample_seed = np.random.randint(0, 10000000000)

        assert isinstance(sample_seed, int)
        self.sampler.sample(arr, corrupt_head, batch, sample_seed)

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

class _BatchElementConverter(object):
    def __init__(self, cuda_enabled=False):
        self.cuda_enabled = cuda_enabled

    def __call__(self, x):
        x = Variable(torch.from_numpy(x))
        if self.cuda_enabled:
            x = x.cuda()
        return x

def convert_triple_tuple_to_torch(batch, config):
    converter = _BatchElementConverter(config.enable_cuda)
    return tuple(map(converter, batch))

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

_SAFE_MINIMAL_BATCH_SIZE = 1

def create_dataloader(triple_source, config, dataset_type=DatasetType.TRAINING):
    """Creates dataloader with certain types"""
    dataset = TripleIndexesDataset(triple_source, dataset_type)

    # Use those C++ extension is fast but then we can't use spawn method to start data loader.
    if dataset_type == DatasetType.TRAINING:
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
    else: # Validation and Test
        data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=min(_SAFE_MINIMAL_BATCH_SIZE, int(config.batch_size*config.evaluation_load_factor)),
            num_workers=config.num_workers,
            pin_memory=True, # May cause system froze because of of non-preemption
            collate_fn=TripleTileCollate(config, triple_source),
        )
    return data_loader

def reciprocal_rank_fn(rank):
    return 1.0/rank

class HitsReducer(object):
    """Used with accumulation function"""

    def __init__(self, target):
        self.target = target

    def __call__(self, value, rank):
        return value + 1 if rank <= self.target else value

def dict_key_gen(*args):
    return "_".join(args)

MEAN_RECIPROCAL_RANK_FEATURE_KEY = 'mean_reciprocal_rank'
MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY = 'mean_filtered_reciprocal_rank'
MEAN_RANK_FEATURE_KEY = 'mean_rank'
MEAN_FILTERED_RANK_FEATURE_KEY = 'mean_filtered_rank'
HITS_1_FEATURE_KEY = 'hits_1'
HITS_1_FILTERED_FEATURE_KEY = "hits_1_filtered"
HITS_3_FEATURE_KEY = 'hits_3'
HITS_3_FILTERED_FEATURE_KEY = 'hits_3_filtered'
HITS_5_FEATURE_KEY = 'hits_5'
HITS_5_FILTERED_FEATURE_KEY = "hits_5_filtered"
HITS_10_FEATURE_KEY = 'hits_10'
HITS_10_FILTERED_FEATURE_KEY = "hits_10_filtered"
HEAD_KEY = "head"
TAIL_KEY = "tail"
RELATION_KEY = "relation"

def _add_rank_statistics(result, key, add, value_fn, *value_args):
    if not add:
        return
    result[key] = value_fn(*value_args)

def _calc_rank(ranks, num_ranks):
    return sum(ranks) / num_ranks

def _calc_reciprocal_rank(ranks, num_ranks):
    return sum(map(reciprocal_rank_fn, ranks)) / num_ranks

def _calc_hits(ranks, target, num_ranks):
    return functools.reduce(HitsReducer(target), ranks) / num_ranks

def get_rank_statistics(rank_list, filtered_rank_list, features, num_entry):
    result = {}
    num_ranks = len(rank_list)

    _calc_reciprocal_rank = lambda ranks, num_ranks: sum(map(reciprocal_rank_fn, ranks)) / num_ranks

    _add_rank_statistics(result, MEAN_RECIPROCAL_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_RECIPROCAL_RANK & features, _calc_reciprocal_rank, rank_list, num_ranks)
    _add_rank_statistics(result, MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK & features, _calc_reciprocal_rank, filtered_rank_list, num_ranks)
    _add_rank_statistics(result, MEAN_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_RANK & features, _calc_rank, rank_list, num_ranks)
    _add_rank_statistics(result, MEAN_FILTERED_RANK_FEATURE_KEY, LinkPredictionStatistics.MEAN_FILTERED_RANK & features, _calc_rank, filtered_rank_list, num_ranks)
    _add_rank_statistics(result, HITS_1_FEATURE_KEY, LinkPredictionStatistics.HITS_1 & features, _calc_hits, rank_list, 1, num_entry)
    _add_rank_statistics(result, HITS_3_FEATURE_KEY, LinkPredictionStatistics.HITS_3 & features, _calc_hits, rank_list, 3, num_entry)
    _add_rank_statistics(result, HITS_5_FEATURE_KEY, LinkPredictionStatistics.HITS_5 & features, _calc_hits, rank_list, 5, num_entry)
    _add_rank_statistics(result, HITS_10_FEATURE_KEY, LinkPredictionStatistics.HITS_10 & features, _calc_hits, rank_list, 10, num_entry)
    _add_rank_statistics(result, HITS_1_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_1_FILTERED & features, _calc_hits, filtered_rank_list, 1, num_entry)
    _add_rank_statistics(result, HITS_3_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_3_FILTERED & features, _calc_hits, filtered_rank_list, 3, num_entry)
    _add_rank_statistics(result, HITS_5_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_5_FILTERED & features, _calc_hits, filtered_rank_list, 5, num_entry)
    _add_rank_statistics(result, HITS_10_FILTERED_FEATURE_KEY, LinkPredictionStatistics.HITS_10_FILTERED & features, _calc_hits, filtered_rank_list, 10, num_entry)

    return result


