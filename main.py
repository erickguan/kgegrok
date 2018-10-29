import torch
import data

def cli():
    data_dir = "data/YAGO3-10"
    dataset = data.TripleIndexesDataset(data_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size)
    for i_batch, sample_batched in enumerate(data_loader):
        print(i_batch, sample_batched)

if __name__ == '__main__':
    cli()

