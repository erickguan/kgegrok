import unittest
import data
import kgekit
from torchvision import transforms

class DataTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        cls.triple_dir = 'tests/fixtures/triples'
        cls.source = data.TripleSource(cls.triple_dir, 'hrt', ' ')
        cls.dataset = data.TripleIndexesDataset(cls.source)

    def test_triple_source(self):
        self.assertEqual(len(self.source.train_set), 4)
        self.assertEqual(len(self.source.valid_set), 1)
        self.assertEqual(len(self.source.test_set), 1)

    def test_dataset(self):
        self.assertEqual(len(self.dataset), 4)
        self.assertEqual(self.dataset[0], kgekit.TripleIndex(0, 0, 1))

    def test_ordered_triple_transform(self):
        transform_dataset = data.TripleIndexesDataset(self.source, transform=transforms.Compose([
            data.OrderedTripleTransform("hrt")
        ]))
        self.assertEqual(transform_dataset[0], [0, 0, 1])
