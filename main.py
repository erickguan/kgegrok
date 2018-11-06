import torch
import data
import kgekit
import models

class Config(object):
    data_dir = "data/YAGO3-10"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 1
    negative_relation = 1
    batch_size = 100
    num_workers = 2
    entity_embedding_dimension = 50
    margin = 0.01
    epoches = 1
    test_head = True
    test_relation = False
    test_tail = True


def train_and_validate(config, model_klass):
    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    data_loader = data.create_dataloader(triple_source, config)
    valid_data_loader = data.create_dataloader(triple_source, config, data.DatasetType.VALIDATION)
    model = model_klass(triple_source, config)

    for i_epoch in range(config.epoches):
        for i_batch, sample_batched in enumerate(data_loader):
            batch, negative_batch = sample_batched
            model.forward(batch, negative_batch)

        for i_batch, sample_batched in enumerate(valid_data_loader):
            # triple has shape (1, 3). We need to tile it for the test.
            for triple in sample_batched:
                if config.test_head:
                    batch = data.expand_triple_to_sets(triple[0, :], triple_source.num_entity, data.TripleElement.HEAD)
                    batch = data.convert_triple_tuple_to_torch(batch)
                    predicted_batch = model.predict(batch)
                    print(predicted_batch, len(predicted_batch))
                    exit()

    return model


def cli():
    config = Config()
    train_and_validate(config, models.TransE)

if __name__ == '__main__':
    cli()

