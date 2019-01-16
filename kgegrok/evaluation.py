import logging
import sys
import threading
import contextlib
import copy
from contextlib import contextmanager

import torch
import torch.multiprocessing as mp

import kgekit
import kgedata
from kgegrok.data import constants
from kgegrok import stats
from kgegrok import data


class AtomicCounter(object):
    """An atomic, thread-safe incrementing counter. Used for passing counter.
    """

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        # Not needed because of GIL.
        try:
            nc = getattr(contextlib, 'nullcontext')
        except AttributeError:
            nc = contextlib.suppress()
        self._lock = nc  # threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

    def decrement(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value -= num
            return self.value

    def reset(self):
        with self._lock:
            self.value = 0


def evaluate_single_triple(prediction, prediction_type, triple_index, config,
                           entities, relations):
    """Returns a list of top `config.report_num_preiction_interactively` results by name."""
    sorted_prediction, indices = torch.sort(prediction)
    prediction_list = []
    if prediction_type == constants.HEAD_KEY or prediction_type == constants.TAIL_KEY:
        for i in range(config.report_num_prediction_interactively):
            if i < len(indices):
                index = indices[i]
                prediction_list.append((index, entities[index]))
    elif prediction_type == constants.RELATION_KEY:
        for i in range(config.report_num_prediction_interactively):
            if i < len(indices):
                prediction_list.append((index, relations[index]))
    else:
        raise RuntimeError(
            "Unknown prediction type {}.".format(prediction_type))

    return predict_links


def _evaluate_prediction_view(result_view, triple_index, rank_fn, datatype):
    """Evaluation on a view of batch."""
    rank, filtered_rank = rank_fn(result_view, triple_index)
    return (datatype, rank, filtered_rank)


def _evaluation_worker_loop(resource, input_q, output):
    ranker = resource.ranker
    try:
        while True:
            p = input_q.get()
            if p is None:
                continue
            if isinstance(p, str) and p == 'STOP':
                raise StopIteration
            batch_tensor, batch, splits = p
            predicted_batch = batch_tensor.data.numpy()
            result = []
            for triple_index, split in zip(batch, splits):
                if split[0] != split[1] and split[1] != split[2]:
                    result.append(
                        _evaluate_prediction_view(
                            predicted_batch[split[0]:split[1]], triple_index,
                            ranker.rank_head, constants.HEAD_KEY))
                    result.append(
                        _evaluate_prediction_view(
                            predicted_batch[split[1]:split[2]], triple_index,
                            ranker.rank_tail, constants.TAIL_KEY))
                if split[2] != split[3]:
                    result.append(
                        _evaluate_prediction_view(
                            predicted_batch[split[2]:split[3]], triple_index,
                            ranker.rank_relation, constants.RELATION_KEY))
            output.put(result)
    except StopIteration:
        print("[Evaluation Worker {}] stops.".format(mp.current_process().name))
        sys.stdout.flush()


def _evaluation_result_thread_loop(resource, output, results_list, counter):
    hr, fhr, tr, ftr, rr, frr = results_list
    RESULTS_LIST = {
        constants.HEAD_KEY: (hr, fhr),
        constants.TAIL_KEY: (tr, ftr),
        constants.RELATION_KEY: (rr, frr)
    }

    try:
        while True:
            results = output.get()
            if results is None:
                continue
            if isinstance(results, str) and results == 'STOP':
                raise StopIteration
            for datatype, rank, filtered_rank in results:
                rank_list, filtered_rank_list = RESULTS_LIST[datatype]
                rank_list.append(rank)
                filtered_rank_list.append(filtered_rank)
            counter.decrement()
    except StopIteration:
        print("[Result Worker {}] stops.".format(mp.current_process().name))
        sys.stdout.flush()


RESULT_LIST_SIZE = 6


class EvaluationProcessPool(object):

    def __init__(self, config, triple_source, context):
        self._config = config
        self._context = context
        self._manager = self._context.Manager()
        self._ns = self._manager.Namespace()
        self._ns.ranker = kgedata.Ranker(triple_source.train_set,
                                        triple_source.valid_set,
                                        triple_source.test_set)
        self._input = self._context.SimpleQueue()
        self._output = self._context.SimpleQueue()
        self._results_list = [list() for _ in range(RESULT_LIST_SIZE)]
        self._counter = AtomicCounter()

    def _prepare_list(self):
        self._counter.reset()
        for i in range(RESULT_LIST_SIZE):
            self._results_list[i].clear()

    def start(self):
        self._prepare_list()
        self._processes = [
            self._context.Process(
                target=_evaluation_worker_loop,
                args=(
                    self._ns,
                    self._input,
                    self._output,
                )) for _ in range(self._config.num_evaluation_workers)
        ]
        self._result_thread = threading.Thread(
            target=_evaluation_result_thread_loop,
            args=(
                self._ns,
                self._output,
                self._results_list,
                self._counter,
            ))
        for p in self._processes:
            p.start()
        self._result_thread.start()

    def stop(self):
        for p in self._processes:
            try:
                p.close()
            except AttributeError:
                p.terminate()  # close() added in 3.7
        # Put as many as stop markers for workers to stop.
        for i in range(self._config.num_evaluation_workers * 2):
            self._input.put('STOP')
        self._output.put('STOP')
        self._output.put('STOP')
        for p in self._processes:
            p.join()
        self._result_thread.join()

    def evaluate_batch(self, test_package):
        """Batch is a Tensor."""
        self._input.put(test_package)
        self._counter.increment()
        logging.debug(
            "Putting a new batch for evaluation. Now we have sent {} batches.".
            format(self._counter.value))

    def wait_evaluation_results(self):
        logging.debug("Starts to wait for result batches.")
        # Protected by GIL
        while self._counter.value > 0:
            logging.debug("counter is now at {}.".format(self._counter.value))
            continue
        else:
            logging.debug("results list {}".format(self._results_list))
            # deep copy that before we destroyed them
            results = tuple([copy.deepcopy(r) for r in self._results_list])

        # Reset
        self._prepare_list()
        logging.debug("results list[0] copied {}".format(results))

        return results


def predict_links(model, triple_source, config, data_loader, pool):
    model.eval()

    for sample_batched in data_loader:
        sampled, batch, splits = sample_batched
        predicted_batch = model.forward(sampled).cpu()
        logging.debug("Current batch's shape {}.".format(predicted_batch.shape))

        pool.evaluate_batch((predicted_batch, batch, splits))

    # Synchonized point. We want all our results back.
    head_ranks, filtered_head_ranks, tail_ranks, filtered_tail_ranks, relation_ranks, filtered_relation_ranks = pool.wait_evaluation_results(
    )
    logging.info(
        "Batch size of rank lists (hr, frr, tr, ftr, rr, frr): {}, {}, {}, {}, {}, {}"
        .format(
            len(head_ranks), len(filtered_head_ranks), len(tail_ranks),
            len(filtered_tail_ranks), len(relation_ranks),
            len(filtered_relation_ranks)))
    return (head_ranks, filtered_head_ranks), (
        tail_ranks, filtered_tail_ranks), (relation_ranks,
                                           filtered_relation_ranks)


# FIXME: can't be used with multiprocess now. See predict_links
def evaulate_prediction_np_collate(model, triple_source, config, ranker,
                                   data_loader):
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
            triple_index = kgedata.TripleIndex(*triple[0, :])

            if (config.report_dimension &
                    stats.StatisticsDimension.SEPERATE_ENTITY) or (
                        config.report_dimension &
                        stats.StatisticsDimension.COMBINED_ENTITY):
                _evaluate_predict_element(
                    model, config, triple_index, triple_source.num_entity,
                    data.TripleElement.HEAD, ranker.rank_head, head_ranks,
                    filtered_head_ranks)
                _evaluate_predict_element(
                    model, config, triple_index, triple_source.num_entity,
                    data.TripleElement.TAIL, ranker.rank_tail, tail_ranks,
                    filtered_tail_ranks)
            if config.report_dimension & stats.StatisticsDimension.RELATION:
                _evaluate_predict_element(
                    model, config, triple_index, triple_source.num_relation,
                    data.TripleElement.RELATION, ranker.rank_relation,
                    relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (
        tail_ranks, filtered_tail_ranks), (relation_ranks,
                                           filtered_relation_ranks)


# FIXME: Not sure what to deal with this old method
def _evaluate_predict_element(model, config, triple_index, num_expands,
                              element_type, rank_fn, ranks_list,
                              filtered_ranks_list):
    """Evaluation a single triple with expanded sets."""
    batch = data.expand_triple_to_sets(
        kgekit.data.unpack(triple_index), num_expands, element_type)
    batch = data.convert_triple_tuple_to_torch(batch, config)
    logging.debug(element_type)
    logging.debug("Batch len: " + str(len(batch)) + "; batch sample: " +
                  str(batch[0]))
    predicted_batch = model.forward(batch).cpu()
    logging.debug("Predicted batch len" + str(len(predicted_batch)) +
                  "; batch sample: " + str(predicted_batch[0]))
    rank, filtered_rank = rank_fn(predicted_batch.data.numpy(), triple_index)
    logging.debug("Rank :" + str(rank) + "; Filtered rank length :" +
                  str(filtered_rank))
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

@contextmanager
def validation_resource_manager(mode, config, triple_source, required_modes=['train_validate', 'test']):
    """prepare resources if validation is needed."""
    enabled = mode in required_modes
    if enabled:
        ctx = mp.get_context('spawn')
        pool = EvaluationProcessPool(config, triple_source, ctx)
        try:
            pool.start()
            yield pool
        finally:
            pool.stop()
    else:
        yield None
