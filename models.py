import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np


class Model(nn.Module):
    """A minimal model interface for other models."""

    def __init__(self, triple_source, config):
        super(Model, self).__init__()
        self.triple_source = triple_source
        self.config = config

    def predict(self, triple):
        raise NotImplementedError

    def forward(self, batch, negative_batch):
        """Args:
        batch tensor with shape (batch_size, 1, 3).
        negative_batch tensor with shape (batch_size, negative_samples, 3).
        To use Embbeddings like (batch_size, embedding_size), we need to extract
        h, r, t into (batch_size)
        """
        raise NotImplementedError


class TransE(Model):
    '''TransE is the first model to introduce translation-based embedding,
    which interprets relations as the translations operating on entities.
    '''

    def __init__(self, triple_source, config):
        super(TransE, self).__init__(triple_source, config)
        self.embedding_dimension = self.config.entity_embedding_dimension
        self.entity_embeddings = nn.Embedding(self.triple_source.num_entity, self.embedding_dimension)
        self.relation_embeddings = nn.Embedding(self.triple_source.num_relation, self.embedding_dimension)

        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)

    def _calc(self,h,t,r):
        return torch.abs(h + r - t)

    # margin-based loss
    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        y = Variable(torch.Tensor([-1]))
        if torch.cuda.is_available():
            y = y.cuda()
        loss = criterion(p_score, n_score, y)
        return loss

    def forward(self, batch, negative_batch):
        pos_h, pos_r, pos_t = batch
        neg_h, neg_r, neg_t = negative_batch
        p_h = self.entity_embeddings(pos_h)
        p_t = self.entity_embeddings(pos_t)
        p_r = self.relation_embeddings(pos_r)
        n_h = self.entity_embeddings(neg_h)
        n_t = self.entity_embeddings(neg_t)
        n_r = self.relation_embeddings(neg_r)
        _p_score = self._calc(p_h, p_t, p_r)
        _n_score = self._calc(n_h, n_t, n_r)
        _p_score = _p_score.view(-1, 1, self.embedding_dimension)
        _n_score = _n_score.view(-1, self.config.negative_entity + self.config.negative_relation, self.embedding_dimension)
        p_score = torch.sum(torch.mean(_p_score, 1), 1)
        n_score = torch.sum(torch.mean(_n_score, 1), 1)
        loss = self.loss_func(p_score, n_score)
        return loss

    def predict(self, batch):
        h, r, t = batch
        p_h = self.entity_embeddings(h)
        p_r = self.relation_embeddings(r)
        p_t = self.entity_embeddings(t)
        _p_score = self._calc(p_h, p_t, p_r)
        p_score = torch.sum(_p_score, 1)
        return p_score.cpu()
