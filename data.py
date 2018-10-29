"""Data loading and processing"""

import kgekit.io

class TripleIndexesDataset(Dataset):
    """Loads triple indexes dataset."""

    def __init__(self, data_dir, triple_order="hrt", delimeter=' ', transform=None):
        """
        Args:
            data_dir (string): Directory with {train,valid,test}.txt.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.trains = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "train.txt"), triple_order, delimeter)
        self.valids = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "valid.txt"), triple_order, delimeter)
        self.tests = kgekit.io.read_triple_indexes(os.path.join(self.data_dir, "test.txt"), triple_order, delimeter)

    def __len__(self):
        return len(self.trains)

    def __getitem__(self, idx):
        sample =  self.trains[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
