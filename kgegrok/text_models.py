import logging
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable

from kgegrok.models import Model


class ComplExText(Model):
    """ComplExText builds more dimension size."""

    def __init__(self, triple_source, config):
        super(ComplEx, self).__init__(triple_source, config)
        self.embedding_dimension = self.config.entity_embedding_dimension

        self.ent_re_embeddings = nn.Embedding(self.triple_source.num_entity,
                                              self.embedding_dimension)
        self.ent_im_embeddings = nn.Embedding(self.triple_source.num_entity,
                                              self.embedding_dimension)
        self.rel_re_embeddings = nn.Embedding(self.triple_source.num_relation,
                                              self.embedding_dimension)
        self.rel_im_embeddings = nn.Embedding(self.triple_source.num_relation,
                                              self.embedding_dimension)
        self.softplus = nn.Softplus()

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def _calc(self, e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im):
        """score function of ComplEx"""
        return torch.sum(
            r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t +
            r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, 1, False)

    def loss_func(self, loss, regul):
        return loss + self.config.lambda_ * regul

    def forward(self, batch):
        pos, neg, labels = batch

        pos_h, pos_r, pos_t = pos.transpose(0, 1)
        e_re_h = self.ent_re_embeddings(pos_h)
        e_im_h = self.ent_im_embeddings(pos_h)
        e_re_t = self.ent_re_embeddings(pos_t)
        e_im_t = self.ent_im_embeddings(pos_t)
        r_re = self.rel_re_embeddings(pos_r)
        r_im = self.rel_im_embeddings(pos_r)

        # Calculating loss to get what the framework will optimize
        if neg is not None:
            neg_h, neg_r, neg_t = neg.view(-1, 3).transpose(0, 1)
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

            res = self._calc(e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im)
            tmp = self.softplus(-labels * res)

            loss = torch.mean(tmp)
            regul = (torch.mean(e_re_h**2) +
                torch.mean(e_im_h**2) +
                torch.mean(e_re_t**2) +
                torch.mean(e_im_t**2) +
                torch.mean(r_re**2) +
                torch.mean(r_im**2))

            return self.loss_func(loss, regul)
        else:
            score = -self._calc(e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im)
            return score
