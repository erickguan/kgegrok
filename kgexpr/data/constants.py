"""Data type and constants"""

from enum import Enum, Flag
from typing import List
from kgedata import TripleIndex

TripleIndexList = List[TripleIndex]

NUM_POSITIVE_INSTANCE = 1

TRIPLE_LENGTH = 3
HEAD_KEY = "head"
TAIL_KEY = "tail"
ENTITY_KEY = "entity"
RELATION_KEY = "relation"

DatasetType = Enum('DatasetType', 'TRAINING VALIDATION TESTING')

DEFAULT_BATCH_SIZE = 100


class TripleElement(Flag):
    HEAD = 1
    RELATION = 2
    TAIL = 3

    @classmethod
    def has_value(cls, value):
        return any(value == item for item in cls)
