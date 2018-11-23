"""Data function for statistics."""

import functools


def reciprocal_rank_fn(rank):
    return 1.0 / rank


class HitsReducer(object):
    """Used with accumulation function"""

    def __init__(self, target):
        self.target = target

    def __call__(self, value, rank):
        return value + 1 if rank <= self.target else value


def dict_key_gen(*args):
    return "_".join(args)


def calc_rank(ranks, num_ranks):
    if len(ranks) != num_ranks:
        raise RuntimeError(
            "Rank are not enough for num_ranks. len(ranks): {}, num_ranks: {}".
            format(len(ranks), num_ranks))
    return sum(ranks) / float(num_ranks)


def calc_reciprocal_rank(ranks, num_ranks):
    if len(ranks) != num_ranks:
        raise RuntimeError(
            "Rank are not enough for num_ranks. len(ranks): {}, num_ranks: {}".
            format(len(ranks), num_ranks))
    return sum(map(reciprocal_rank_fn, ranks)) / float(num_ranks)


def calc_hits(target, ranks, num_ranks):
    if len(ranks) != num_ranks:
        raise RuntimeError(
            "Rank are not enough for num_entry. len(ranks): {}, num_ranks: {}".
            format(len(ranks), num_ranks))
    return functools.reduce(HitsReducer(target), ranks) / float(num_ranks)
