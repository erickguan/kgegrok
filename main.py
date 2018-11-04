import torch
import data
from torchvision import transforms
import kgekit

def cli():
    data_dir = "data/YAGO3-10"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 1
    negative_relation = 1
    batch_size = 100
    num_workers = 2

    triple_source = data.TripleSource(data_dir, triple_order, triple_delimiter)
    dataset = data.TripleIndexesDataset(triple_source)
    negative_sampler = kgekit.LCWANoThrowSampler(triple_source.train_set, negative_entity, negative_relation, kgekit.LCWANoThrowSamplerStrategy.Hash)
    corruptor = kgekit.BernoulliCorruptor(triple_source.train_set)

    data_loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=transforms.Compose([
            data.BernoulliCorruptionCollate(triple_source, corruptor),
            data.LCWANoThrowCollate(triple_source, negative_sampler),
        ]))
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, len(sample_batched), type(sample_batched), sample_batched[0])
        if i_batch > 2:
            exit()

if __name__ == '__main__':
    cli()

