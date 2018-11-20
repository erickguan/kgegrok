import pprint
import data
import kgekit
import numpy as np
import logging
from collections import defaultdict
import json


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

def _evaluate_prediction_view(result_view, triple_index, rank_fn, ranks_list, filtered_ranks_list):
    """Evaluation on a view of batch."""
    rank, filtered_rank = rank_fn(result_view, triple_index)
    ranks_list.append(rank)
    filtered_ranks_list.append(filtered_rank)

def gen_drawer_option(config, title=None):
    if title is not None:
        title = "{}/{}".format(config.name, title)
    return dict(fillarea=True, xlabel="Epoch", width=800, height=800, title=title)

def _report_prediction_element(element):
    pprint.pprint(element)

def _common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def _append_drawer(drawer, epoch, result, prefix_key=None):
    assert epoch is not None
    for key, value in result.items():
        drawer_key = data.dict_key_gen(prefix_key, key) if prefix_key is not None else key
        drawer.append(drawer_key, X=np.array([epoch], dtype='f'), Y=np.array([value], dtype='f'))

def report_prediction_result(triple_source, config, result, printing=True, epoch=None, drawer=None):
    heads, tails, relations = result
    ret_values = {}

    if config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY:
        head_result = data.get_evaluation_statistics(*heads, config.report_features, triple_source.num_entity)
        tail_result = data.get_evaluation_statistics(*tails, config.report_features, triple_source.num_entity)
        ret_values[data.HEAD_KEY] = head_result
        ret_values[data.TAIL_KEY] = tail_result
        _report_prediction_element(head_result)
        _report_prediction_element(tail_result)
        if drawer is not None:
            _append_drawer(drawer, epoch, head_result, data.HEAD_KEY)
            _append_drawer(drawer, epoch, tail_result, data.TAIL_KEY)

    elif config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY:
        combined = {k: (h + t) / 2.0 for k, h, t in _common_entries(data.get_evaluation_statistics(*heads, config.report_features, triple_source.num_entity), data.get_evaluation_statistics(*tails, config.report_features, triple_source.num_entity))}
        ret_values[data.ENTITY_KEY] = combined
        _report_prediction_element(combined)
        if drawer is not None:
            _append_drawer(drawer, epoch, combined)

    if config.report_dimension & data.StatisticsDimension.RELATION:
        relation_result = data.get_evaluation_statistics(*relations, config.report_features, triple_source.num_relation)
        ret_values[data.RELATION_KEY] = relation_result
        _report_prediction_element(relation_result)
        if drawer is not None:
            _append_drawer(drawer, epoch, relation_result, data.RELATION_KEY)
    return ret_values

def evaulate_prediction(model, triple_source, config, ranker, data_loader):
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
        predicted_batch = model.forward(sampled).cpu().data.numpy()

        # TODO: ranker can be extended to multiprocessing for more performance.
        for triple_index, split in zip(batch, splits):
            if split[0] != split[1] and split[1] != split[2]:
                _evaluate_prediction_view(predicted_batch[split[0]:split[1]], triple_index, ranker.rankHead, head_ranks, filtered_head_ranks)
                _evaluate_prediction_view(predicted_batch[split[1]:split[2]], triple_index, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if split[2] != split[3]:
                _evaluate_prediction_view(predicted_batch[split[2]:split[3]], triple_index, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

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

            if (config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY) or (config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY):
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.HEAD, ranker.rankHead, head_ranks, filtered_head_ranks)
                _evaluate_predict_element(model, config, triple_index, triple_source.num_entity, data.TripleElement.TAIL, ranker.rankTail, tail_ranks, filtered_tail_ranks)
            if config.report_dimension & data.StatisticsDimension.RELATION:
                _evaluate_predict_element(model, config, triple_index, triple_source.num_relation, data.TripleElement.RELATION, ranker.rankRelation, relation_ranks, filtered_relation_ranks)

    return (head_ranks, filtered_head_ranks), (tail_ranks, filtered_tail_ranks), (relation_ranks, filtered_relation_ranks)

class ReportDrawer(object):
    """It's actually expecting Visdom drawer."""

    def __init__(self, drawer, config):
        self.drawer = drawer
        # holds the our key: visdom window key
        self.plots = {}
        self.plots_opts = {}
        self.config = config

    def _is_plot_exist(self, key):
        return key in self.plots_opts

    def _lazy_create_plot(self, key, X, Y):
        """Return False if creation is not needed."""
        if key not in self.plots:
            self.plots[key] = self.drawer.line(X=X, Y=Y, opts=self.plots_opts[key])
            return True
        return False

    def append(self, key, X, Y):
        """X and Y are numpy array"""
        if not self._is_plot_exist:
            raise RuntimeError("{} doesn't exists in drawer.".format(key))
        if not self._lazy_create_plot(key, X, Y):
            # append condition
            self.drawer.line(X=X, Y=Y, win=self.plots[key], update='append')

    def create_plot(self, key, options=None):
        # lazy creation so we can avoid an empty record in the beginning
        self.plots_opts[key] = options

    def _dump_win_data(self, win):
        content = self.drawer.get_window_data(win)
        if content is None or len(content) == 0:
            content = "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(content)
        return content

    def dump_raw_data(self):
        """Dumps all raw data."""

        data = []
        for _, win in self.plots.items():
            data.append(_dump_win_data(win))
        return data


DRAWING_KEY_AND_CONDITION = {
data.StatisticsDimension.COMBINED_ENTITY: [
    (data.MEAN_RANK_FEATURE_KEY, data.LinkPredictionStatistics.MEAN_RANK),
    (data.MEAN_FILTERED_RANK_FEATURE_KEY, data.LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.MEAN_RECIPROCAL_RANK_FEATURE_KEY, data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY, data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.HITS_1_FEATURE_KEY, data.LinkPredictionStatistics.HITS_1),
    (data.HITS_3_FEATURE_KEY, data.LinkPredictionStatistics.HITS_3),
    (data.HITS_5_FEATURE_KEY, data.LinkPredictionStatistics.HITS_5),
    (data.HITS_10_FEATURE_KEY, data.LinkPredictionStatistics.HITS_10),
    (data.HITS_1_FILTERED_FEATURE_KEY, data.LinkPredictionStatistics.HITS_1_FILTERED),
    (data.HITS_3_FILTERED_FEATURE_KEY, data.LinkPredictionStatistics.HITS_3_FILTERED),
    (data.HITS_5_FILTERED_FEATURE_KEY, data.LinkPredictionStatistics.HITS_5_FILTERED),
    (data.HITS_10_FILTERED_FEATURE_KEY, data.LinkPredictionStatistics.HITS_10_FILTERED),
],
data.StatisticsDimension.SEPERATE_ENTITY: [
    (data.dict_key_gen(data.HEAD_KEY, data.MEAN_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.HEAD_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.HEAD_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.HEAD_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_1_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_3_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_5_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_10_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_1_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_3_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_5_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.HEAD_KEY, data.HITS_10_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10_FILTERED),

    (data.dict_key_gen(data.TAIL_KEY, data.MEAN_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.TAIL_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.TAIL_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.TAIL_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_1_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_3_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_5_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_10_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_1_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_3_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_5_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.TAIL_KEY, data.HITS_10_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10_FILTERED),
],
data.StatisticsDimension.RELATION: [
    (data.dict_key_gen(data.RELATION_KEY, data.MEAN_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RANK),
    (data.dict_key_gen(data.RELATION_KEY, data.MEAN_FILTERED_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RANK),
    (data.dict_key_gen(data.RELATION_KEY, data.MEAN_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_RECIPROCAL_RANK),
    (data.dict_key_gen(data.RELATION_KEY, data.MEAN_FILTERED_RECIPROCAL_RANK_FEATURE_KEY), data.LinkPredictionStatistics.MEAN_FILTERED_RECIPROCAL_RANK),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_1_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_3_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_5_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_10_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_1_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_1_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_3_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_3_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_5_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_5_FILTERED),
    (data.dict_key_gen(data.RELATION_KEY, data.HITS_10_FILTERED_FEATURE_KEY), data.LinkPredictionStatistics.HITS_10_FILTERED),
]
}

def prepare_plot_validation_result(drawer, config):
    if config.report_dimension & data.StatisticsDimension.COMBINED_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[data.StatisticsDimension.COMBINED_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    elif config.report_dimension & data.StatisticsDimension.SEPERATE_ENTITY:
        for key, condition in DRAWING_KEY_AND_CONDITION[data.StatisticsDimension.SEPERATE_ENTITY]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))
    if config.report_dimension & data.StatisticsDimension.RELATION:
        for key, condition in DRAWING_KEY_AND_CONDITION[data.StatisticsDimension.RELATION]:
            if condition & config.report_features:
                drawer.create_plot(key, gen_drawer_option(config, key))

    return drawer
