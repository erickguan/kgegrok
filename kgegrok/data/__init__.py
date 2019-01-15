"""Data loading and processing"""

import os
import kgekit.io
import kgekit.utils
import kgedata
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision.transforms import Compose
import functools
from kgegrok.data import constants, transformers
from kgegrok.utils import deprecation
import kgegrok.utils


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
        self._train_set, num_failed = kgekit.io.read_triple_indexes_numpy(
            os.path.join(self.data_dir, self.TRAIN_FILENAME),
            triple_order=triple_order,
            delimiter=delimiter)
        assert num_failed == 0
        self._valid_set, num_failed = kgekit.io.read_triple_indexes_numpy(
            os.path.join(self.data_dir, self.VALID_FILENAME),
            triple_order=triple_order,
            delimiter=delimiter)
        assert num_failed == 0
        self._test_set, num_failed = kgekit.io.read_triple_indexes_numpy(
            os.path.join(self.data_dir, self.TEST_FILENAME),
            triple_order=triple_order,
            delimiter=delimiter)
        assert num_failed == 0

        self._collect_stats()

    def _collect_stats(self):
        max_train = np.amax(self._train_set, axis=0)
        max_valid = np.amax(self._valid_set, axis=0)
        max_test = np.amax(self._test_set, axis=0)
        max_nums = np.amax(np.stack([max_train, max_valid, max_test], axis=0), axis=0)
        self._num_entity = max(max_nums[0], max_nums[2]) + 1
        self._num_relation = max_nums[1] + 1

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

class SequentialBatchSampler(object):
    """generates an item of i."""

    def __init__(self, dataset):
        self.len = len(dataset)

    def __iter__(self):
        self._iter = iter(range(self.len))
        return self

    def __next__(self):
        return [next(self._iter)]

def flat_collate_fn(batch):
    """Flatten pytorch dataloader list since we only load 1 batch for dataloader."""
    return batch[0]

MIN_NUM_PAD = 1

class TripleDataset(Dataset):
    """Base class for triple dataset."""

    def __init__(self,
                 triples,
                 batch_size=constants.DEFAULT_BATCH_SIZE,
                 drop_last=False,
                 pad_batch=True,
                 transform=None):
        """Builds the dataset with common parameters and data."""
        self._pad_batch = pad_batch
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._transform = transform
        self._build_batches(triples)

    def _build_batches(self, triples):
        self._data = []

        for i in range(0, len(triples), self._batch_size):
            v = np.array(triples[i:i+self._batch_size], dtype=np.int64)
            self._data.append(v)

        if self._drop_last and len(self._data) > 1 and self._data[-1].shape != self._data[-2].shape:
            self._data.pop()
        elif self._pad_batch:
            num_pads = self._data[-1].shape[0] % max(MIN_NUM_PAD, kgegrok.utils.num_cuda_devices())
            while num_pads > 0:
                for i in range(len(self._data)):
                    pad = self._data[i][:num_pads]
                    num_pads -= pad.shape[0]
                    self._data[-1] = np.concatenate([self._data[-1], pad])

    def __len__(self):
        """Returns the number of batches for triples."""
        return len(self._data)

    def __getitem__(self, idx):
        """returns the batch with transform."""
        sample = self._data[idx]
        if self._transform:
            sample = self._transform(sample)

        return sample

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform

def get_triples_from_batch(batch):
    """Returns h, r, t and possible label from batch."""
    deprecation("Not tested anymore", since="0.3.0")

    multiple_samples = (batch.ndim == 3)
    if multiple_samples:
        batch_size, num_samples, num_element = batch.shape
        elements = np.split(batch, num_element, axis=2)
        return (e.reshape(batch_size, num_samples) for e in elements)
    else:
        batch_size, num_element = batch.shape
        elements = np.split(batch, num_element, axis=1)
        return (e.reshape(batch_size) for e in elements)


class _BatchElementConverter(object):
    def __init__(self, cuda_enabled=False):
        deprecation("Not tested anymore", since="0.3.0")
        self.cuda_enabled = cuda_enabled

    def __call__(self, x):
        return np_to_tensor(x, self.cuda_enabled)


def convert_triple_tuple_to_torch(batch, config, enable_cuda_override=None):
    deprecation("Not tested anymore", since="0.3.0")
    if enable_cuda_override is not None:
        converter = _BatchElementConverter(enable_cuda_override)
    else:
        converter = _BatchElementConverter(config.enable_cuda)
    return tuple(map(converter, batch))


def expand_triple_to_sets(triple, num_expands, arange_target):
    """Tiles triple into a large sets for testing. One node will be initialized with arange.
    Returns (h, r, t), each with a shape of (num_expands,)
    """
    deprecation("Not tested anymore", since="0.3.0")

    if not constants.TripleElement.has_value(arange_target):
        raise RuntimeError(
            "arange_target is set to wrong value. It has to be one of TripleElement but it's "
            + str(arange_target))
    h, r, t = triple

    if arange_target == constants.TripleElement.HEAD:
        h = np.arange(num_expands, dtype=np.int64)
        r = np.tile(np.array([r], dtype=np.int64), num_expands)
        t = np.tile(np.array([t], dtype=np.int64), num_expands)
    elif arange_target == constants.TripleElement.RELATION:
        h = np.tile(np.array([h], dtype=np.int64), num_expands)
        r = np.arange(num_expands, dtype=np.int64)
        t = np.tile(np.array([t], dtype=np.int64), num_expands)
    elif arange_target == constants.TripleElement.TAIL:
        h = np.tile(np.array([h], dtype=np.int64), num_expands)
        r = np.tile(np.array([r], dtype=np.int64), num_expands)
        t = np.arange(num_expands, dtype=np.int64)
    else:
        raise RuntimeError(
            "Miracle happened. arange_target passed the validation and reached impossible branch."
        )

    return (h, r, t)


_SAFE_MINIMAL_BATCH_SIZE = 1


SEED_OFFSET = 100

def create_dataloader_from_dataset(dataset, config):
    """Returns dataloader with dataset."""

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=config.num_workers,
        pin_memory=True,
        timeout=config.batch_worker_timeout,
        batch_sampler=SequentialBatchSampler(dataset),
        collate_fn=flat_collate_fn
    )

def create_dataloader(triple_source,
                      config,
                      build_label=False,
                      dataset_type=constants.DatasetType.TRAINING):
    """Creates dataloader.
    When training, returns dataloader using perturbation sampler with bernoulli trick for training.
    When validation/testing, returns dataloader with tile support.
    If other configurations are used, such dataloader needs to be created manually.
    See create_dataloader_from_dataset.
    """

    # Use those C++ extension is fast but then we can't use spawn method to start data loader.
    if dataset_type == constants.DatasetType.TRAINING:
        corruptor = kgedata.BernoulliCorruptor(
            triple_source.train_set,
            triple_source.num_relation,
            config.negative_entity,
            config.base_seed + SEED_OFFSET)
        negative_sampler = kgedata.PerturbationSampler(
                    triple_source.train_set,
                    triple_source.num_entity,
                    triple_source.num_relation,
                    config.negative_entity,
                    config.negative_relation,
                    config.base_seed + 2*SEED_OFFSET,
                    kgedata.PerturbationSamplerStrategy.Hash)

        transforms = [
            transformers.CorruptionFlagGenerator(corruptor),
            transformers.NegativeBatchGenerator(negative_sampler),
        ]
        if build_label:
            transforms.append(transformers.LabelBatchGenerator(config))
        else:
            transforms.append(transformers.none_label_batch_generator)
        transforms.append(transformers.tensor_transform)

        dataset = TripleDataset(
            triple_source.train_set,
            batch_size=config.batch_size,
            transform=Compose(transforms))
    else:  # Validation and Test
        batch_size = max(_SAFE_MINIMAL_BATCH_SIZE,
                         int(config.batch_size * config.evaluation_load_factor))
        transforms = [
            transformers.TripleTileGenerator(config, triple_source),
            transformers.test_batch_transform
        ]
        if dataset_type == constants.DatasetType.VALIDATION:
            triple_set = triple_source.valid_set
        else:
            triple_set = triple_source.test_set

        dataset = TripleDataset(
            triple_set,
            batch_size=batch_size,
            transform=Compose(transforms))

    return create_dataloader_from_dataset(dataset, config)


def sieve_and_expand_triple(triple_source, entities, relations, head, relation,
                            tail):
    """Tile on a unknown element. returns a tuple of size 3 with h, r, t."""
    deprecation("Not tested anymore", since="0.3.0")

    batch_size, num_samples, num_element = batch.shape
    elements = np.split(batch, num_element, axis=2)
    # return (e.reshape(batch_size) for e in elements)

    if head == '?':
        r = relations[relation]
        t = entities[tail]
        triple_index = kgedata.TripleIndex(-1, r, t)

        h = np.arange(triple_source.num_entity, dtype=np.int64)
        r = np.tile(np.array([r], dtype=np.int64), triple_source.num_entity)
        t = np.tile(np.array([t], dtype=np.int64), triple_source.num_entity)
        prediction_type = constants.HEAD_KEY
    elif relation == '?':
        h = entities[head]
        t = entities[tail]
        triple_index = kgedata.TripleIndex(h, -1, t)

        h = np.tile(np.array([h], dtype=np.int64), triple_source.num_relation)
        r = np.arange(triple_source.num_relation, dtype=np.int64)
        t = np.tile(np.array([t], dtype=np.int64), triple_source.num_relation)
        prediction_type = constants.RELATION_KEY
    elif tail == '?':
        r = relations[relation]
        h = entities[head]
        triple_index = kgedata.TripleIndex(h, r, -1)

        h = np.tile(np.array([h], dtype=np.int64), triple_source.num_entity)
        r = np.tile(np.array([r], dtype=np.int64), triple_source.num_entity)
        t = np.arange(triple_source.num_entity, dtype=np.int64)
        prediction_type = constants.TAIL_KEY
    else:
        raise RuntimeError("head, relation, tail are known.")

    return (h, r, t), prediction_type, triple_index


# class LiteralCollate(object):
#     def __init__(self):
#             self.source,
#             negative_sampler,
#             literals=['facts'],
#             sample_negative_for_non_triples=False,
#             transforms=dict(
#                 triple_transform=data.OrderedTripleListTransform("hrt"),
#                 fact_transform=data.FactTransform(),
#             ),
#         )(self.samples, 0)
