"""Includes a bunch of transformer for dataset."""

import numpy as np
from kgexpr.data import constants
import kgekit


class CorruptionFlagGenerator(object):
    """Generates corrupted head/tail decision in uniform distribution.
    if used with bernoulli corruptor, it's Bernoulli Distribution based on tph.
    True means we will corrupt head."""

    def __init__(self, corruptor):
        self.corruptor = corruptor

    def __call__(self, batch):
        """return corruption flag and actual batch"""
        choices = self.corruptor.make_random_choice(batch)
        return choices, batch


class NegativeBatchGenerator(object):
    """Process and sample an negative batch. Supports multiprocess from pytorch.
    We don't throw if |set(subject, predict)| is empty.
    Returns:
        Positive tensor with shape (batch_size, 1, 3).
        Negative tensor with shape (batch_size, negative_samples, 3).
    """

    def __init__(self, negative_sampler):
        self.sampler = negative_sampler

    def __call__(self, batch_set):
        corrupt_head, batch = batch_set
        batch_size = batch.shape[0]
        negative_batch = self.sampler.sample(corrupt_head, batch)
        return batch, negative_batch

def label_batch_generator(sample):
    """Add data label for (batch, negative_batch).
    positive batch shape: (batch_size, 3)
    negative batch shape: (batch_size*negative_samples, 3).
    label batch shape: (batch_size*(1+negative_samples),).
    """
    batch, negative_batch = sample

    labels_shape = (batch.shape[0]*(1+negative_batch.shape[1]),)
    labels = np.full(labels_shape, -1, dtype=np.int64)
    labels[:batch.shape[0]] = 1
    return batch, negative_batch, labels

def batch_transpose_transform(sample):
    batch, negative_batch, labels = sample
    batch = batch.T
    negative_batch = negative_batch.T

    return batch, negative_batch, labels

def _np_to_tensor(x, cuda_enabled):
    x = torch.from_numpy(x)
    # Input is an index to find relevant embeddings. We don't track them
    x.requires_grad_(False)
    if cuda_enabled:
        x = x.cuda()
    return x

def _batch_element_to_tensor(x, cuda_enabled):
    if x is None:
        return x
    return _np_to_tensor(x, cuda_enabled)

class TensorTransform(object):
    """Returns batch, negative_batch, labels by the tensor."""

    def __init__(self, config, enable_cuda_override=None):
        if enable_cuda_override is not None:
            self.cuda_enabled = enable_cuda_override
        else:
            self.cuda_enabled = config.enable_cuda

    def __call__(self, sample):
        return tuple(map(lambda x: _batch_element_to_tensor(x, self.cuda_enabled), sample))


def label_prediction_collate(sample):
    """Add all positive labels for sample.
    """
    tiled, batch, splits = sample

    labels_shape = (tiled.shape[0])
    labels = np.full(labels_shape, 1, dtype=np.int64)

    return tiled, batch, splits, labels

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

class BreakdownCollator(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        batch, negative_batch, labels = sample
        if labels is not None:
            labels = data.np_to_tensor(labels, self.config.enable_cuda)
        batch = data.convert_triple_tuple_to_torch(
            data.get_triples_from_batch(batch), self.config, False)
        negative_batch = data.convert_triple_tuple_to_torch(
            data.get_triples_from_batch(negative_batch), self.config, False)
        return batch, negative_batch, labels

def none_label_collate(sample):
    """Adds a None to collators."""
    batch, negative_batch = sample
    return (batch, negative_batch, None)


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
        sampled, splits = kgedata.expand_triple_batch(
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
        batch = np.empty((batch_size, constants.TRIPLE_LENGTH), dtype=np.int64)
        for i in range(batch_size):
            for o in self.triple_order:
                t = samples[i]
                if o == 'h':
                    batch[i, 0] = t.head
                elif o == 'r':
                    batch[i, 1] = t.relation
                elif o == 't':
                    batch[i, 2] = t.tail

        return batch


class FactTransform(object):
    """Returns a list of fact transformed. For example bert.

    Args:
        triple_order (str): Desired triple order in list.
    """

    def __init__(self, triple_order):
        kgekit.utils.assert_triple_order(triple_order)
        self.triple_order = triple_order

    def __call__(self, samples):
        batch_size = len(samples)
        batch = np.empty((batch_size, constants.NUM_POSITIVE_INSTANCE,
                          constants.TRIPLE_LENGTH),
                         dtype=np.int64)
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
