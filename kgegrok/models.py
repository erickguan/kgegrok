"""Model module."""

import logging
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable


class Model(nn.Module):
  """A minimal model interface for other models."""

  @classmethod
  def require_labels(cls):
    return False

  def __init__(self, triple_source, config):
    super(Model, self).__init__()
    self.triple_source = triple_source
    self.config = config

  def forward(self, batch):
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
    self.entity_embeddings = nn.Embedding(self.triple_source.num_entity,
                                          self.embedding_dimension)
    self.relation_embeddings = nn.Embedding(self.triple_source.num_relation,
                                            self.embedding_dimension)

    self.register_parameter(
        'y', torch.nn.Parameter(torch.tensor([-1.0]), requires_grad=False))
    self.criterion = nn.MarginRankingLoss(self.config.margin, False)

    nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
    nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

  def _calc(self, h, t, r):
    return torch.abs(h + r - t)

  def loss_func(self, p_score, n_score):
    return self.criterion(p_score, n_score, self.y)

  def forward(self, batch):
    pos, neg, _ = batch

    pos_h, pos_r, pos_t = pos.transpose(0, 1)
    p_h = self.entity_embeddings(pos_h)
    p_t = self.entity_embeddings(pos_t)
    p_r = self.relation_embeddings(pos_r)
    _p_score = self._calc(p_h, p_t, p_r)

    if neg is not None:
      neg_h, neg_r, neg_t = neg.view(-1, 3).transpose(0, 1)
      n_h = self.entity_embeddings(neg_h)
      n_t = self.entity_embeddings(neg_t)
      n_r = self.relation_embeddings(neg_r)
      _n_score = self._calc(n_h, n_t, n_r)
      _n_score = _n_score.view(
          -1, self.config.negative_entity + self.config.negative_relation,
          self.embedding_dimension)

      n_score = torch.sum(torch.mean(_n_score, 2), 1)
      p_score = torch.mean(_p_score, 1)
      loss = self.loss_func(p_score, n_score)
      return loss
    else:
      p_score = torch.sum(_p_score, 1)
      return p_score


class ComplEx(Model):
  """ComplEx builds more dimension size."""

  @classmethod
  def require_labels(cls):
    return True

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
        r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t
        - r_im * e_im_h * e_re_t, 1, False)

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
      regul = (
          torch.mean(e_re_h**2) + torch.mean(e_im_h**2) + torch.mean(e_re_t**2)
          + torch.mean(e_im_t**2) + torch.mean(r_re**2) + torch.mean(r_im**2))

      return self.loss_func(loss, regul)
    else:
      score = -self._calc(e_re_h, e_im_h, e_re_t, e_im_t, r_re, r_im)
      return score

class ConvE(Model):
  """ConvE builds with the convolution operation."""

  @classmethod
  def require_labels(cls):
    return True

  def __init__(self, triple_source, config):
    super(ConvE, self).__init__(triple_source, config)
    self.embedding_dimension = self.config.entity_embedding_dimension

    self.emb_e = torch.nn.Embedding(self.triple_source.num_entity, self.embedding_dimension)
    self.emb_rel = torch.nn.Embedding(self.triple_source.num_relation, self.embedding_dimension)
    self.inp_drop = torch.nn.Dropout(self.config.input_dropout)
    self.hidden_drop = torch.nn.Dropout(self.config.dropout)
    self.feature_map_drop = torch.nn.Dropout2d(self.config.feature_map_dropout)
    self.loss = torch.nn.BCELoss()

    self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.config.use_bias)
    self.bn0 = torch.nn.BatchNorm2d(1)
    self.bn1 = torch.nn.BatchNorm2d(32)
    self.bn2 = torch.nn.BatchNorm1d(self.embedding_dimension)
    self.register_parameter('b', torch.nn.Parameter(torch.zeros(self.triple_source.num_entity)))
    self.fc = torch.nn.Linear(int(362.88 * self.embedding_dimension), self.embedding_dimension)
    self.sigmoid = torch.nn.Sigmoid()

    nn.init.xavier_uniform_(self.emb_e.weight.data)
    nn.init.xavier_uniform_(self.emb_rel.weight.data)

  def forward(self, batch):
    pos, neg, labels = batch

    if neg is not None:
      h, r, t = neg.view(-1, 3).transpose(0, 1)
      _, labels = labels
    else:
      h, r, t = pos.transpose(0, 1)

    e1_embedded = self.emb_e(h).view(-1, 1, 10, 20)
    rel_embedded = self.emb_rel(r).view(-1, 1, 10, 20)
    stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)
    stacked_inputs = self.bn0(stacked_inputs)
    x = self.inp_drop(stacked_inputs)
    x = self.conv1(x)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.feature_map_drop(x)
    x = x.view(self.config.batch_size, -1)
    x = self.fc(x)
    x = self.hidden_drop(x)
    x = self.bn2(x)
    x = F.relu(x)
    x = torch.mm(x, self.emb_e.weight.transpose(1,0))
    x += self.b.expand_as(x)
    pred = self.sigmoid(x)

    if neg is not None:
      print(pred.shape, h.shape, labels.shape)
      labels = ((1.0 - self.config.label_smoothing_epsilon)*labels) + (1.0/labels.size(1))
      return self.loss(pred, labels)
    else:
      return pred
