"""different data collator function"""

import numpy as np
from kgexpr.data import constants
from kgexpr.stats.constants import StatisticsDimension
import kgekit


def _make_random_choice(size, probability):
    if isinstance(probability[0], float):
        choices = [
            np.random.choice([True, False], p=probability) for _ in range(size)
        ]
    else:
        choices = [
            np.random.choice([True, False], p=probability[i])
            for i in range(size)
        ]
    return choices


def _round_probablities(probability):
    return (probability, 1 - probability)


class BernoulliCorruptionCollate(object):
    """Generates corrupted head/tail decision in Bernoulli Distribution based on tph.
    True means we will corrupt head."""

    def __init__(self, triple_source, bernoulli_corruptor):
        """Loads the data from source."""
        self.train_set = triple_source.train_set
        self.bernoulli_corruptor = bernoulli_corruptor

    def __call__(self, batch: constants.TripleIndexList):
        """return corruption flag and actual batch"""
        if isinstance(batch[0], list):
            probabilities = [
                _round_probablities(
                    self.bernoulli_corruptor.getProbablityRelation(t[1]))
                for t in batch
            ]
        else:
            probabilities = [
                _round_probablities(
                    self.bernoulli_corruptor.getProbablityRelation(t.relation))
                for t in batch
            ]
        return _make_random_choice(len(batch), probabilities), batch


class UniformCorruptionCollate(object):
    """Generates corrupted head/tail decision in uniform distribution.
    True means we will corrupt head."""

    def __call__(self, batch: constants.TripleIndexList):
        """batch is a list"""
        return _make_random_choice(len(batch), [0.5, 0.5]), batch


class NumpyCollate(object):
    """Process triples and put them into a triple index.
    Returns:
        Positive tensor with shape (batch_size, 1, 3).
    """

    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, batch: constants.TripleIndexList):
        """process a mini-batch."""
        batch_size = len(batch)
        if self.transform:
            batch = self.transform(batch)
        batch = np.array(batch, dtype=np.int64)[:, np.newaxis, :]
        return batch


def label_prediction_collate(sample):
    """Add all positive labels for sample.
    """
    tiled, batch, splits = sample

    labels_shape = (tiled.shape[0])
    labels = np.full(labels_shape, 1.0, dtype=np.float32)

    return tiled, batch, splits, labels

def label_collate(sample):
    """Add data label for (batch, negative_batch).
    positive batch shape: (batch_size, 1, 3)
    negative batch shape: (batch_size, negative_samples, 3).
    label batch shape: (batch_size*(1+negative_samples),).
    """
    batch, negative_batch = sample

    labels_shape = (batch.shape[0], 1+negative_batch.shape[1], 1)
    labels = np.full(labels_shape, -1.0, dtype=np.float32)
    labels[:,0,:] = 1.0
    new_labels_shape = (labels_shape[0] * labels_shape[1])
    np.reshape(labels, new_labels_shape)

    return batch, negative_batch, labels


class LCWANoThrowCollate(object):
    """Process and sample an negative batch. Supports multiprocess from pytorch.
    We don't throw if |set(subject, predict)| is empty.
    Returns:
        Positive tensor with shape (batch_size, 1, 3).
        Negative tensor with shape (batch_size, negative_samples, 3).
    """

    def __init__(self, triple_source, negative_sampler, transform=None):
        self.sampler = negative_sampler
        self.transform = transform

    def __call__(self, batch_set, sample_seed=None):
        if isinstance(batch_set, tuple) and len(batch_set) == 2:
            corrupt_head, batch = batch_set
        else:
            raise RuntimeError(
                "Wrong parameter type. Need a tuple (corrupt_head, batch). Got "
                + str(type(batch_set)))
        batch_size = len(batch)
        negative_batch = np.empty(
            (batch_size, self.sampler.numNegativeSamples(),
             constants.TRIPLE_LENGTH),
            dtype=np.int64)
        if sample_seed is None:
            sample_seed = np.random.randint(0, 10000000000)

        assert isinstance(sample_seed, int)
        self.sampler.sample(negative_batch, corrupt_head, batch, sample_seed)

        # We don't want to waste memory here. so batch was passed as a list
        if self.transform:
            batch = self.transform(batch)
        return batch, negative_batch


class TripleTileCollate(object):
    """Process triples and put them into a tiled but flat numpy array.
    Args:
        config: config object for reading dimension information
        triple_source: triple source function
    Returns:
        Positive tensor with shape (batch_size * varied_size, 1, 3).
            varied_size will depends on testing dimension, num_entity and num_relation.
        Original batch from PyTorch,
        Splits split points
    """

    def __init__(self, config, triple_source):
        self.config = config
        self.triple_source = triple_source

    def __call__(self, batch: constants.TripleIndexList):
        """process a mini-batch."""
        sampled, splits = kgekit.expand_triple_batch(
            batch, self.triple_source.num_entity,
            self.triple_source.num_relation,
            (self.config.report_dimension & StatisticsDimension.SEPERATE_ENTITY)
            or (self.config.report_dimension &
                StatisticsDimension.COMBINED_ENTITY),
            self.config.report_dimension & StatisticsDimension.RELATION)

        sampled = sampled[:, np.newaxis, :]
        return (sampled, batch, splits)

class LiteralCollate(object):
    def __init__(self, source, negative_sampler, literals, transforms, sample_negative_for_non_triples=False):
        self.source = source
        self.negative_sampler = negative_sampler
        self.literals = literals
        self.sample_negative_for_non_triples = sample_negative_for_non_triples
        self.transforms = transforms
