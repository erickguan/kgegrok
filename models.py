"""Model module."""

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable
import numpy as np
import logging


class Model(nn.Module):
    """A minimal model interface for other models."""

    def __init__(self, triple_source, config):
        super(Model, self).__init__()
        self.triple_source = triple_source
        self.config = config

    def forward(self, batch, negative_batch=None):
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

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

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

    def forward(self, batch, negative_batch=None):
        pos_h, pos_r, pos_t = batch
        p_h = self.entity_embeddings(pos_h)
        p_t = self.entity_embeddings(pos_t)
        p_r = self.relation_embeddings(pos_r)
        _p_score = self._calc(p_h, p_t, p_r)
        _p_score = _p_score.view(-1, 1, self.embedding_dimension)
        logging.debug("_p_score shape " + str(_p_score.shape))

        if negative_batch is not None:
            neg_h, neg_r, neg_t = negative_batch
            n_h = self.entity_embeddings(neg_h)
            n_t = self.entity_embeddings(neg_t)
            n_r = self.relation_embeddings(neg_r)
            _n_score = self._calc(n_h, n_t, n_r)
            _n_score = _n_score.view(-1, self.config.negative_entity + self.config.negative_relation, self.embedding_dimension)
            n_score = torch.sum(torch.mean(_n_score, 1), 1)

            p_score = torch.sum(torch.mean(_p_score, 1), 1)
            loss = self.loss_func(p_score, n_score)
            return loss
        else:
            p_score = torch.sum(_p_score, (1, 2))
            logging.debug("p_score shape " + str(p_score.shape))
            return p_score


class ComplEx(Model):
    """ComplEx builds more dimension size."""

    def __init__(self, triple_source, config):
        super(ComplEx, self).__init__(triple_source, config)
        self.embedding_dimension = self.config.entity_embedding_dimension

        self.ent_re_embeddings = nn.Embedding(self.triple_source.num_entity, self.embedding_dimension)
        self.ent_im_embeddings = nn.Embedding(self.triple_source.num_entity, self.embedding_dimension)
        self.rel_re_embeddings = nn.Embedding(self.triple_source.num_relation, self.embedding_dimension)
        self.rel_im_embeddings = nn.Embedding(self.triple_source.num_relation, self.embedding_dimension)
        self.softplus = nn.Softplus().cuda()

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im):
        """score function of ComplEx"""
        return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, 1, False)

    def loss_func(self, loss, regul):
        return loss + self.config.lambda_ * regul

    def forward(self, batch, negative_batch):
        pos_h, pos_r, pos_t = batch
        y = np.tile(np.array([-1], dtype=np.int64), pos_h.shape[0])
        e_re_h = self.ent_re_embeddings(pos_h)
        e_im_h = self.ent_im_embeddings(pos_h)
        e_re_t = self.ent_re_embeddings(pos_t)
        e_im_t = self.ent_im_embeddings(pos_t)
        r_re = self.rel_re_embeddings(pos_r)
        r_im = self.rel_im_embeddings(pos_r)

        # Calculating loss to get what the framework will optimize
        if negative_batch is not None:
            neg_h, neg_r, neg_t = negative_batch
            neg_y = np.tile(np.array([-1], dtype=np.int64), neg_h.shape[0])
            neg_e_re_h = self.ent_re_embeddings(neg_h)
            neg_e_im_h = self.ent_im_embeddings(neg_h)
            neg_e_re_t = self.ent_re_embeddings(neg_t)
            neg_e_im_t = self.ent_im_embeddings(neg_t)
            neg_r_re = self.rel_re_embeddings(neg_r)
            neg_r_im = self.rel_im_embeddings(neg_r)

            e_re_h = torch.cat((e_re_h, neg_e_re_h))
            e_im_h = torch.cat((e_im_h, neg_e_im_h))
            e_re_t = torch.cat((e_re_t, neg_e_re_t))
            e_im_t = torch.cat((e_im_t, neg_e_im_t))
            r_re = torch.cat((r_re, neg_r_re))
            r_im = torch.cat((r_im, neg_r_im))
            y = torch.cat((y, neg_y))

            res = self._calc(e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im)
            tmp = self.softplus(-y * res)
            loss = torch.mean(tmp)
            regul = torch.mean(e_re_h**2) + torch.mean(e_im_h**2) + torch.mean(e_re_t**2) + torch.mean(e_im_t**2) + torch.mean(r_re**2) + torch.mean(r_im**2)
            loss = self.loss_func(loss, regul)

            return loss
        else:
            score = -self._calc(p_re_h, p_im_h, p_re_t, p_im_t, p_re_r, p_im_r)
            return score
