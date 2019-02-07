"""Asynchorous evaluation. It aims to utilize GPUs when the CPU is busy.
"""
import logging
import sys
import threading
import queue
import atexit
from contextlib import contextmanager

import torch
import torch.multiprocessing as mp

import kgekit
import kgedata
from kgegrok.data import constants
from kgegrok import stats
from kgegrok import data


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


def _evaluation_worker_loop(evaluator):
    results_list = evaluator._results_list
    try:
        while True:
            p = evaluator._input.get()
            if p is None:
                continue
            if isinstance(p, str) and p == 'STOP':
                raise StopIteration
            batch_tensor, batch, splits = p
            predicted_batch = batch_tensor.data.numpy()
            results = evaluator._ranker.submit(predicted_batch, batch, splits, ascending_rank=True)
            for result in results:
                results_list['hr'].append(result[0])
                results_list['fhr'].append(result[1])
                results_list['tr'].append(result[2])
                results_list['ftr'].append(result[3])
                results_list['rr'].append(result[4])
                results_list['frr'].append(result[5])
            evaluator._cv.acquire()
            evaluator._counter -= 1
            evaluator._cv.notify()
            evaluator._cv.release()
    except StopIteration:
        print("[Evaluation Worker {}] stops.".format(threading.current_thread().name))
        sys.stdout.flush()


_CV_TIMEOUT = 0.01

# Note: might be worthy to invest on concurrent.future. Doesn't feel like thread pool helps much.
class ParallelEvaluator(object):
    """Evaluates the validation/test batch parallelly."""

    def __init__(self, config, triple_source):
        self._config = config
        self._ranker = kgedata.Ranker(triple_source.train_set,
                                      triple_source.valid_set,
                                      triple_source.test_set)
        self._input = queue.SimpleQueue()
        self._cv = threading.Condition()
        self._results_list = {
            'hr': [],
            'fhr': [],
            'tr': [],
            'ftr': [],
            'rr': [],
            'frr': []
        }
        self._prepare_list()

        self._threads = [
            threading.Thread(
                target=_evaluation_worker_loop,
                args=(self,),
            ) for _ in range(self._config.num_evaluation_workers)
        ]
        for p in self._threads:
            p.start()
        atexit.register(self.cleanup)


    def _prepare_list(self):
        self._counter = 0
        self._results_list['hr'] = []
        self._results_list['fhr'] = []
        self._results_list['tr'] = []
        self._results_list['ftr'] = []
        self._results_list['rr'] = []
        self._results_list['frr'] = []

    def cleanup(self):
        # Put as many as stop markers for workers to stop.
        for i in range(self._config.num_evaluation_workers * 2):
            self._input.put('STOP')
        for p in self._threads:
            p.join()

    def evaluate_batch(self, test_package):
        """Batch is a Tensor."""
        logging.debug(
            "Putting a new batch {} for evaluation. Now we have sent {} batches.".
            format(test_package, self._counter))
        self._input.put(test_package)
        self._counter += 1

    def get_results(self):
        logging.debug("Starts to wait for result batches.")

        while True:
            self._cv.acquire()
            self._cv.wait(timeout=_CV_TIMEOUT)
            logging.debug("counter is now at {}.".format(self._counter))
            if self._counter <= 0:
                self._cv.release()
                break
            self._cv.release()

        logging.debug("results list {}".format(self._results_list))
        # deep copy that before we destroyed them

        results = tuple([
            self._results_list['hr'],
            self._results_list['fhr'],
            self._results_list['tr'],
            self._results_list['ftr'],
            self._results_list['rr'],
            self._results_list['frr'],
        ])

        # Reset
        self._prepare_list()
        logging.debug("results list[0] copied {}".format(results))

        return results


def predict_links(model, triple_source, config, data_loader, evaluator):
    model.eval()

    for sample_batched in data_loader:
        sampled, batch, splits = sample_batched
        predicted_batch = model.forward(sampled).cpu()
        logging.debug("Current batch's shape {}.".format(predicted_batch.shape))

        evaluator.evaluate_batch((predicted_batch, batch, splits))

    # Synchonized point. We want all our results back.
    return evaluator.get_results()

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
def validation_resource_manager(config, triple_source, required_modes=['train_validate', 'test']):
    """prepare resources if validation is needed."""
    enabled = config.mode in required_modes
    if enabled:
        ctx = mp.get_context('spawn')
        pool = ParallelEvaluator(config, triple_source, ctx)
        try:
            pool.start()
            yield pool
        finally:
            pool.stop()
    else:
        yield None
