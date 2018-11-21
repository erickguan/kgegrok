import torch.multiprocessing as mp
import kgekit
import data
import stats

def _evaluate_prediction_view(output, result_view, triple_index, rank_fn, datatype):
    """Evaluation on a view of batch."""
    rank, filtered_rank = rank_fn(result_view, triple_index)
    output.put((datatype, rank, filtered_rank))

def _evaluation_worker_loop(resource, input, output):
    ranker = resource.ranker
    try:
        for batch_tensor, triple_index, splits in iter(input.get, 'STOP'):
            batch = batch_tensor.data.numpy()
            for triple_index, split in zip(batch, splits):
                if split[0] != split[1] and split[1] != split[2]:
                    _evaluate_prediction_view(output, predicted_batch[split[0]:split[1]], triple_index, ranker.rankHead, data.HEAD_KEY)
                    _evaluate_prediction_view(output, predicted_batch[split[1]:split[2]], triple_index, ranker.rankTail, data.TAIL_KEY)
                if split[2] != split[3]:
                    _evaluate_prediction_view(output, predicted_batch[split[2]:split[3]], triple_index, ranker.rankRelation, data.RELATION_KEY)
    except StopIteration:
        print("Evaluation worker {} stops.".format(mp.current_process().name))

class EvaluationProcessPool(object):
    def __init__(self, config, triple_source, context):
        self._config = config
        self._context = context
        self._manager = mp.Manager()
        self._ns = self._manager.Namespace()
        self._ns.ranker = kgekit.Ranker(triple_source.train_set, triple_source.valid_set, triple_source.test_set)
        self._input = mp.SimpleQueue()
        self._output = mp.SimpleQueue()
        self._counter = 0

    def start(self):
        self._processes = [
            self._context.Process(
                target=_evaluation_worker_loop,
                args=(self._ns, self._input, self._output)
            )
            for _ in range(self._config.evaluation_workers)
        ]
        for p in self._processes:
            p.start()

    def stop(self):
        for p in self._processes:
            p.close()
        # Put as many as stop markers for workers to stop.
        for i in range(self._config.evaluation_workers * 2):
            self._input.put('STOP')
        for p in self._processes:
            p.join()

    def evaluate_batch(self, batch):
        """Batch is a Tensor."""
        self._input.put(batch)
        self._counter += 1

    def wait_evaluation_results(self, hr, fhr, tr, ftr, rr, frr):
        RESULTS_LIST = {
            data.HEAD_KEY: (hr, fhr),
            data.TAIL_KEY: (tr, ftr),
            data.RELATION_KEY: (rr, frr)
        }

        while self._counter <= 0:
            datatype, rank, filtered_rank = self.output.get()
            self._counter -= 1
            rank_list, filtered_rank_list = RESULTS_LIST[datatype]
            rank_list.append(rank)
            filtered_rank_list.append(filtered_rank)


def predict_links(model, triple_source, config, data_loader, pool):
    model.eval()

    head_ranks = []
    filtered_head_ranks = []
    tail_ranks = []
    filtered_tail_ranks = []
    relation_ranks = []
    filtered_relation_ranks = []

    for i_batch, sample_batched in enumerate(data_loader):
        sampled, batch, splits = sample_batched
        sampled = data.convert_triple_tuple_to_torch(data.get_triples_from_batch(sampled), config)
        predicted_batch = model.forward(sampled).cpu()

        pool.evaluate_batch(predicted_batch)

    # Synchonized point. We want all our results back.
    pool.wait_evaluation_results(head_ranks, filtered_head_ranks, tail_ranks, filtered_tail_ranks, relation_ranks, filtered_relation_ranks)
    logger.info("Batch size of rank lists (hr, frr, tr, ftr, rr, frr): {}, {}, {}, {}, {}, {}".format(
        len(head_ranks),
        len(filtered_head_ranks),
        len(tail_ranks),
        len(filtered_tail_ranks),
        len(relation_ranks),
        len(filtered_relation_ranks)
    ))
    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

# FIXME: can't be used with multiprocess now. See predict_links
def evaulate_prediction_np_collate(model, triple_source, config, ranker, data_loader):
    """use with NumpyCollate."""
    model.eval()

    head_ranks = []
    filtered_head_ranks = []
    tail_ranks = []
    filtered_tail_ranks = []
    relation_ranks = []
    filtered_relation_ranks = []

    for i_batch, sample_batched in enumerate(data_loader):
        # sample_batched is a list of triple. triple has shape (1, 3). We need to tile it for the test.
        for triple in sample_batched:
            triple_index = kgekit.TripleIndex(*triple[0, :])

            if (config.report_dimension & stats.StatisticsDimension.SEPERATE_ENTITY) or (config.report_dimension & stats.StatisticsDimension.COMBINED_ENTITY):
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.report_dimension & stats.StatisticsDimension.RELATION:
                _evaluate_predict_element(model, config, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

# FIXME: Not sure what to deal with this old method
def _evaluate_predict_element(model, config, triple_index, num_expands, element_type, rank_fn, ranks_list, filtered_ranks_list):
    """Evaluation a single triple with expanded sets."""
    batch = data.expand_triple_to_sets(kgekit.data.unpack(triple_index), num_expands, element_type)
    batch = data.convert_triple_tuple_to_torch(batch, config)
    logging.debug(element_type)
    logging.debug("Batch len: " + str(len(batch)) + "; batch sample: " + str(batch[0]))
    predicted_batch = model.forward(batch).cpu()
    logging.debug("Predicted batch len" + str(len(predicted_batch)) + "; batch sample: " + str(predicted_batch[0]))
    rank, filtered_rank = rank_fn(predicted_batch.data.numpy(), triple_index)
    logging.debug("Rank :" + str(rank) + "; Filtered rank length :" + str(filtered_rank))
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)
