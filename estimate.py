"""Training function module."""

import data
import kgekit

def evaulate(model, triple_source, config, ranker, data_loader):
    model.eval()

    head_ranks = []
    head_filtered_ranks = []
    tail_ranks = []
    tail_filtered_ranks = []
    relation_ranks = []
    relation_filtered_ranks = []

    for i_batch, sample_batched in enumerate(data_loader):
        # triple has shape (1, 3). We need to tile it for the test.
        for triple in sample_batched:
            if config.test_head:
                batch = data.expand_triple_to_sets(triple[0, :], triple_source.num_entity, data.TripleElement.HEAD)
                batch = data.convert_triple_tuple_to_torch(batch)
                predicted_batch = model.predict(batch)
                triple_index = kgekit.TripleIndex(*triple[0, :])
                rank, filtered_rank = ranker.rankHead(predicted_batch.data.numpy(), triple_index)
                head_ranks.append(rank)
                head_filtered_ranks.append(filtered_rank)
            if config.test_tail:
                batch = data.expand_triple_to_sets(triple[0, :], triple_source.num_entity, data.TripleElement.TAIL)
                batch = data.convert_triple_tuple_to_torch(batch)
                predicted_batch = model.predict(batch)
                triple_index = kgekit.TripleIndex(*triple[0, :])
                rank, filtered_rank = ranker.rankTail(predicted_batch.data.numpy(), triple_index)
                tail_ranks.append(rank)
                tail_filtered_ranks.append(filtered_rank)
            if config.test_relation:
                batch = data.expand_triple_to_sets(triple[0, :], triple_source.num_entity, data.TripleElement.RELATION)
                batch = data.convert_triple_tuple_to_torch(batch)
                predicted_batch = model.predict(batch)
                triple_index = kgekit.TripleIndex(*triple[0, :])
                rank, filtered_rank = ranker.rankRelation(predicted_batch.data.numpy(), triple_index)
                relation_ranks.append(rank)
                relation_filtered_ranks.append(filtered_rank)

    return (head_ranks, head_filtered_ranks), (tail_ranks, tail_filtered_ranks), (relation_ranks, relation_filtered_ranks)

def train_and_validate(config, model_klass):
    # Data loaders have many processes. Here it's a main program.
    triple_source = data.TripleSource(config.data_dir, config.triple_order, config.triple_delimiter)
    data_loader = data.create_dataloader(triple_source, config)
    valid_data_loader = data.create_dataloader(triple_source, config, data.DatasetType.VALIDATION)
    model = model_klass(triple_source, config)
    ranker = kgekit.Ranker(triple_source.train_set, triple_source.valid_set, triple_source.test_set)

    for i_epoch in range(config.epoches):
        model.train()

        for i_batch, sample_batched in enumerate(data_loader):
            batch, negative_batch = sample_batched
            model.forward(batch, negative_batch)

        t = evaulate(model, triple_source, config, ranker, valid_data_loader)
        print(t)

    return model
