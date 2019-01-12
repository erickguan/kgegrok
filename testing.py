from kgexpr import data
from kgexpr import utils
from kgexpr.data import transformers
import torch
import numpy as np
import kgedata
from torchvision.transforms import Compose

config = utils.Config()

triple_source = data.TripleSource("data/YAGO3-10", "hrt", " ")
ds = data.TripleDataset(triple_source.train_set, batch_size=2)
negative_sampler = kgedata.LCWANoThrowSampler(
            triple_source.train_set,
            triple_source.num_entity,
            triple_source.num_relation,
            config.negative_entity,
            config.negative_relation,
            config.base_seed,
            kgedata.LCWANoThrowSamplerStrategy.Hash)
corruptor = kgedata.BernoulliCorruptor(triple_source.train_set, triple_source.num_relation, config.negative_entity, config.base_seed+1)
transforms = [
    transformers.CorruptionFlagGenerator(corruptor),
    transformers.NegativeBatchGenerator(
        triple_source,
        negative_sampler),
]
# collates.append(collators.label_collate)
transform_fn = Compose(transforms)
ds2 = data.TripleDataset(triple_source.train_set, batch_size=2, transform=transform_fn)
dl2 = iter(torch.utils.data.DataLoader(
        ds2,
        batch_sampler=data.sequential_batch_sampler(ds2),
        collate_fn=data.flat_collate_fn,
        num_workers=0,
        pin_memory=True  # May cause system froze because of not enough physical memory
    )
)

for i in range(5):
    batch = next(dl2)
    print(batch[0].shape, batch[1].shape)
