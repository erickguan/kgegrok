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
        self.softplus = nn.Softplus()
        if torch.cuda.is_available():
            self.softplus = self.softplus.cuda()

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

# Untested
# class Analogy(Model):
#     def __init__(self, triple_source, config):
#         super(Analogy, self).__init__(triple_source, config)
# 		self.ent_re_embeddings = nn.Embedding(self.triple_source.num_entity, self.config.entity_embedding_dimension/2)
# 		self.ent_im_embeddings = nn.Embedding(self.triple_source.num_entity, self.config.entity_embedding_dimension/2)
# 		self.rel_re_embeddings = nn.Embedding(self.triple_source.num_relation, self.config.entity_embedding_dimension/2)
# 		self.rel_im_embeddings = nn.Embedding(self.triple_source.num_relation, self.config.entity_embedding_dimension/2)
# 		self.ent_embeddings = nn.Embedding(self.triple_source.num_entity, self.config.entity_embedding_dimension)
# 		self.rel_embeddings = nn.Embedding(self.triple_source.num_relation, self.config.entity_embedding_dimension)
#         self.softplus = nn.Softplus()
#         if torch.cuda.is_available():
# 		    self.softplus = self.softplus.cuda()
# 		nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

# 	def _calc(self, e_re_h, e_im_h, e_h, e_re_t, e_im_t, e_t, r_re, r_im, r):
# 	    """score function of Analogy, which is the hybrid of ComplEx and DistMult"""
# 		return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t, 1 ,False) +
#             torch.sum(e_h*e_t*r, 1, False)

# 	def loss_func(self, loss, regul):
# 		return loss + self.config.lambda_*regul

#     def forward(self, batch, negative_batch):
#         pos_h, pos_r, pos_t = batch
#         y = np.tile(np.array([-1], dtype=np.int64), pos_h.shape[0])
#         e_re_h = self.ent_re_embeddings(pos_h)
#         e_im_h = self.ent_im_embeddings(pos_h)
#         e_re_t = self.ent_re_embeddings(pos_t)
#         e_im_t = self.ent_im_embeddings(pos_t)
#         r_re = self.rel_re_embeddings(pos_r)
#         r_im = self.rel_im_embeddings(pos_r)

#         # Calculating loss to get what the framework will optimize
#         if negative_batch is not None:
#             neg_h, neg_r, neg_t = negative_batch
#             neg_y = np.tile(np.array([-1], dtype=np.int64), neg_h.shape[0])
#             neg_e_re_h = self.ent_re_embeddings(neg_h)
#             neg_e_im_h = self.ent_im_embeddings(neg_h)
#             neg_e_re_t = self.ent_re_embeddings(neg_t)
#             neg_e_im_t = self.ent_im_embeddings(neg_t)
#             neg_r_re = self.rel_re_embeddings(neg_r)
#             neg_r_im = self.rel_im_embeddings(neg_r)
#     def forward:
# 		batch_h, batch_t, batch_r=self.get_all_instance()
# 		batch_y=self.get_all_labels()
# 		e_re_h=self.ent_re_embeddings(batch_h)
# 		e_im_h=self.ent_im_embeddings(batch_h)
# 		e_h = self.ent_embeddings(batch_h)
# 		e_re_t=self.ent_re_embeddings(batch_t)
# 		e_im_t=self.ent_im_embeddings(batch_t)
# 		e_t=self.ent_embeddings(batch_t)
# 		r_re=self.rel_re_embeddings(batch_r)
# 		r_im=self.rel_im_embeddings(batch_r)
# 		r = self.rel_embeddings(batch_r)
# 		y=batch_y
# 		res=self._calc(e_re_h,e_im_h,e_h,e_re_t,e_im_t,e_t,r_re,r_im,r)
# 		loss = torch.mean(self.softplus(- y * res))
# 		regul= torch.mean(e_re_h**2)+torch.mean(e_im_h**2)*torch.mean(e_h**2)+torch.mean(e_re_t**2)+torch.mean(e_im_t**2)+torch.mean(e_t**2)+torch.mean(r_re**2)+torch.mean(r_im**2)+torch.mean(r**2)
# 		#Calculating loss to get what the framework will optimize
# 		loss =  self.loss_func(loss,regul)
# 		return loss
# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_re_h=self.ent_re_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_re_t=self.ent_re_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_re_r=self.rel_re_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_im_h=self.ent_im_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_im_t=self.ent_im_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_im_r=self.rel_im_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_h=self.ent_im_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_t=self.ent_im_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_r=self.rel_im_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_score = -self._calc(p_re_h, p_im_h, p_h, p_re_t, p_im_t, p_t, p_re_r, p_im_r, p_r)
# 		return p_score.cpu()

# # Untested
# class DistMult(Model):
# 	def __init__(self,config):
# 		super(DistMult,self).__init__(config)
# 		self.ent_embeddings=nn.Embedding(self.triple_source.num_entity,self.config.entity_embedding_dimension)
# 		self.rel_embeddings=nn.Embedding(self.triple_source.num_relation,self.config.entity_embedding_dimension)
# 		self.softplus=nn.Softplus().cuda()
# 		self.init_weights()
# 	def init_weights(self):
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
# 	# score function of DistMult
# 	def _calc(self,h,t,r):
# 		return torch.sum(h*t*r,1,False)
# 	def loss_func(self,loss,regul):
# 		return loss+self.config.lmbda*regul
# 	def forward(self):
# 		batch_h,batch_t,batch_r=self.get_all_instance()
# 		batch_y=self.get_all_labels()
# 		e_h=self.ent_embeddings(batch_h)
# 		e_t=self.ent_embeddings(batch_t)
# 		e_r=self.rel_embeddings(batch_r)
# 		y=batch_y
# 		res=self._calc(e_h,e_t,e_r)
# 		tmp=self.softplus(- y * res)
# 		loss = torch.mean(tmp)
# 		regul = torch.mean(e_h ** 2) + torch.mean(e_t ** 2) + torch.mean(e_r ** 2)
# 		#Calculating loss to get what the framework will optimize
# 		loss =  self.loss_func(loss,regul)
# 		return loss
# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_e_h=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_e_t=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_e_r=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_score=-self._calc(p_e_h,p_e_t,p_e_r)
# 		return p_score.cpu()

#         class RESCAL(Model):
# 	def __init__(self,config):
# 		super(RESCAL,self).__init__(config)
# 		self.ent_embeddings=nn.Embedding(self.triple_source.num_entity,self.config.entity_embedding_dimension)
# 		self.rel_matrices=nn.Embedding(self.triple_source.num_relation,self.config.entity_embedding_dimension*self.config.entity_embedding_dimension)
# 		self.init_weights()
# 	def init_weights(self):
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_matrices.weight.data)
# 	# score function of RESCAL
# 	def _calc(self,h,t,r):
# 		return h*torch.matmul(r,t)
# 	# margin-based loss
# 	def loss_func(self,p_score,n_score):
# 		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
# 		y=Variable(torch.Tensor([1])).cuda()
# 		loss=criterion(p_score,n_score,y)
# 		return loss
# 	def forward(self):
# 		pos_h,pos_t,pos_r=self.get_postive_instance()
# 		neg_h,neg_t,neg_r=self.get_negtive_instance()
# 		p_h=self.ent_embeddings(pos_h).view(-1,self.config.entity_embedding_dimension,1)
# 		p_t=self.ent_embeddings(pos_t).view(-1,self.config.entity_embedding_dimension,1)
# 		p_r=self.rel_matrices(pos_r).view(-1,self.config.entity_embedding_dimension,self.config.entity_embedding_dimension)
# 		n_h=self.ent_embeddings(neg_h).view(-1,self.config.entity_embedding_dimension,1)
# 		n_t=self.ent_embeddings(neg_t).view(-1,self.config.entity_embedding_dimension,1)
# 		n_r=self.rel_matrices(neg_r).view(-1,self.config.entity_embedding_dimension,self.config.entity_embedding_dimension)
# 		_p_score = self._calc(p_h, p_t, p_r).view(-1, 1, self.config.entity_embedding_dimension)
# 		_n_score = self._calc(n_h, n_t, n_r).view(-1, 1, self.config.entity_embedding_dimension)
# 		p_score=torch.sum(torch.mean(_p_score,1,False),1)
# 		n_score=torch.sum(torch.mean(_n_score,1,False),1)
# 		loss=self.loss_func(p_score,n_score)
# 		return loss
# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda()).view(-1,self.config.entity_embedding_dimension,1)
# 		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda()).view(-1,self.config.entity_embedding_dimension,1)
# 		p_r_e=self.rel_matrices(Variable(torch.from_numpy(predict_r)).cuda()).view(-1,self.config.entity_embedding_dimension,self.config.entity_embedding_dimension)
# 		p_score=-torch.sum(self._calc(p_h_e, p_t_e, p_r_e),1)
# 		return p_score.cpu()

# # Untested
# class TransD(Model):
# 	def __init__(self,config):
# 		super(TransD,self).__init__(config)
# 		self.ent_embeddings=nn.Embedding(self.triple_source.num_entity,self.config.entity_embedding_dimension)
# 		self.rel_embeddings=nn.Embedding(self.triple_source.num_relation,self.config.entity_embedding_dimension)
# 		self.ent_transfer=nn.Embedding(self.triple_source.num_entity,self.config.entity_embedding_dimension)
# 		self.rel_transfer=nn.Embedding(self.triple_source.num_relation,self.config.entity_embedding_dimension)
# 		self.init_weights()
# 	def init_weights(self):
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.ent_transfer.weight.data)
# 		nn.init.xavier_uniform_(self.rel_transfer.weight.data)
# 	r'''
# 	TransD constructs a dynamic mapping matrix for each entity-relation pair by
# 	considering the diversity of entities and relations simultaneously.
# 	Compared with TransR/CTransR, TransD has fewer parameters and
# 	has no matrix vector multiplication.
# 	'''
# 	def _transfer(self,e,t,r):
# 		return F.normalize(e+torch.sum(e*t,1,True)*r,2,1)
# 	def _calc(self,h,t,r):
# 		return torch.abs(h+r-t)
# 	def loss_func(self,p_score,n_score):
# 		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
# 		y=Variable(torch.Tensor([-1])).cuda()
# 		loss=criterion(p_score,n_score,y)
# 		return loss
# 	def forward(self):
# 		pos_h,pos_t,pos_r=self.get_postive_instance()
# 		neg_h,neg_t,neg_r=self.get_negtive_instance()
# 		p_h_e=self.ent_embeddings(pos_h)
# 		p_t_e=self.ent_embeddings(pos_t)
# 		p_r_e=self.rel_embeddings(pos_r)
# 		n_h_e=self.ent_embeddings(neg_h)
# 		n_t_e=self.ent_embeddings(neg_t)
# 		n_r_e=self.rel_embeddings(neg_r)
# 		p_h_t=self.ent_transfer(pos_h)
# 		p_t_t=self.ent_transfer(pos_t)
# 		p_r_t=self.rel_transfer(pos_r)
# 		n_h_t=self.ent_transfer(neg_h)
# 		n_t_t=self.ent_transfer(neg_t)
# 		n_r_t=self.rel_transfer(neg_r)
# 		p_h=self._transfer(p_h_e,p_h_t,p_r_t)
# 		p_t=self._transfer(p_t_e,p_t_t,p_r_t)
# 		p_r=p_r_e
# 		n_h=self._transfer(n_h_e,n_h_t,n_r_t)
# 		n_t=self._transfer(n_t_e,n_t_t,n_r_t)
# 		n_r=n_r_e
# 		_p_score = self._calc(p_h, p_t, p_r)
# 		_n_score = self._calc(n_h, n_t, n_r)
# 		_p_score = _p_score.view(-1, 1, self.config.entity_embedding_dimension)
# 		_n_score = _n_score.view(-1, self.config.negative_ent + self.config.negative_rel, self.config.entity_embedding_dimension)
# 		p_score=torch.sum(torch.mean(_p_score, 1),1)
# 		n_score=torch.sum(torch.mean(_n_score, 1),1)
# 		loss=self.loss_func(p_score,n_score)
# 		return loss
# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_r_e=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_h_t=self.ent_transfer(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_t_t=self.ent_transfer(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_r_t=self.rel_transfer(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_h=self._transfer(p_h_e,p_h_t,p_r_t)
# 		p_t=self._transfer(p_t_e,p_t_t,p_r_t)
# 		p_r=p_r_e
# 		_p_score = self._calc(p_h, p_t, p_r)
# 		p_score=torch.sum(_p_score,1)
# 		return p_score.cpu()

# # Untested
# class TransH(Model):
# 	def __init__(self,config):
# 		super(TransH,self).__init__(config)
# 		self.ent_embeddings=nn.Embedding(triple_source.num_entity,self.config.entity_embedding_dimension)
# 		self.rel_embeddings=nn.Embedding(triple_source.num_relation,self.config.entity_embedding_dimension)
# 		self.norm_vector=nn.Embedding(triple_source.num_relation,self.config.entity_embedding_dimension)
# 		self.init_weights()
# 	def init_weights(self):
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.norm_vector.weight.data)
# 	r'''
# 	To preserve the mapping propertities of 1-N/N-1/N-N relations,
# 	TransH inperprets a relation as a translating operation on a hyperplane.
# 	'''
# 	def _transfer(self,e,norm):
# 		return e - torch.sum(e * norm, 1, True) * norm
# 	def _calc(self,h,t,r):
# 		return torch.abs(h+r-t)
# 	# margin-based loss
# 	def loss_func(self,p_score,n_score):
# 		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
# 		y=Variable(torch.Tensor([-1])).cuda()
# 		loss=criterion(p_score,n_score,y)
# 		return loss
# 	def forward(self):
# 		pos_h,pos_t,pos_r=self.get_postive_instance()
# 		neg_h,neg_t,neg_r=self.get_negtive_instance()
# 		p_h_e=self.ent_embeddings(pos_h)
# 		p_t_e=self.ent_embeddings(pos_t)
# 		p_r_e=self.rel_embeddings(pos_r)
# 		n_h_e=self.ent_embeddings(neg_h)
# 		n_t_e=self.ent_embeddings(neg_t)
# 		n_r_e=self.rel_embeddings(neg_r)
# 		p_norm=self.norm_vector(pos_r)
# 		n_norm=self.norm_vector(neg_r)

# 		p_h_e=F.normalize(p_h_e,2,1)
# 		p_t_e=F.normalize(p_t_e,2,1)
# 		p_r_e=F.normalize(p_r_e,2,1)
# 		n_h_e=F.normalize(n_h_e,2,1)
# 		n_t_e=F.normalize(n_t_e,2,1)
# 		n_r_e=F.normalize(n_r_e,2,1)

# 		p_norm=F.normalize(p_norm,2,1)
# 		n_norm=F.normalize(n_norm,2,1)

# 		p_h=self._transfer(p_h_e,p_norm)
# 		p_t=self._transfer(p_t_e,p_norm)
# 		p_r=p_r_e
# 		n_h=self._transfer(n_h_e,n_norm)
# 		n_t=self._transfer(n_t_e,n_norm)
# 		n_r=n_r_e
# 		_p_score = self._calc(p_h, p_t, p_r)
# 		_n_score = self._calc(n_h, n_t, n_r)
# 		_p_score = _p_score.view(-1, 1, self.config.entity_embedding_dimension)
# 		_n_score = _n_score.view(-1, self.config.negative_ent + self.config.negative_rel, self.config.entity_embedding_dimension)
# 		p_score=torch.sum(torch.mean(_p_score, 1),1)
# 		n_score=torch.sum(torch.mean(_n_score, 1),1)
# 		loss=self.loss_func(p_score,n_score)
# 		return loss

# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
# 		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
# 		p_r_e=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_norm=self.norm_vector(Variable(torch.from_numpy(predict_r)).cuda())
# 		p_h_e=F.normalize(p_h_e,2,1)
# 		p_t_e=F.normalize(p_t_e,2,1)
# 		p_r_e=F.normalize(p_r_e,2,1)
# 		p_norm=F.normalize(p_norm,2,1)
# 		p_h=self._transfer(p_h_e,p_norm)
# 		p_t=self._transfer(p_t_e,p_norm)
# 		p_r=p_r_e
# 		_p_score=self._calc(p_h, p_t, p_r)
# 		p_score=torch.sum(_p_score,1)
# 		return p_score.cpu()

# # Untested
# class TransR(Model):
# 	def __init__(self,config):
# 		super(TransR,self).__init__(config)
# 		self.ent_embeddings=nn.Embedding(self.triple_source.num_entity,self.config.ent_size)
# 		self.rel_embeddings=nn.Embedding(self.triple_source.num_relation,self.config.rel_size)
# 		self.transfer_matrix=nn.Embedding(self.triple_source.num_relation,self.config.ent_size*self.config.rel_size)
# 		self.init_weights()
# 	def init_weights(self):
# 		nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
# 		nn.init.xavier_uniform_(self.transfer_matrix.weight.data)
# 	def _transfer(self,transfer_matrix,embeddings):
# 		return torch.matmul(transfer_matrix,embeddings)
# 	r'''
# 	TransR first projects entities from entity space to corresponding relation space
# 	and then builds translations between projected entities.
# 	'''
# 	def _calc(self,h,t,r):
# 		return torch.abs(h+r-t)
# 	# margin-based loss
# 	def loss_func(self,p_score,n_score):
# 		criterion= nn.MarginRankingLoss(self.config.margin,False).cuda()
# 		y=Variable(torch.Tensor([-1])).cuda()
# 		loss=criterion(p_score,n_score,y)
# 		return loss
# 	def forward(self):
# 		pos_h,pos_t,pos_r=self.get_postive_instance()
# 		neg_h,neg_t,neg_r=self.get_negtive_instance()
# 		p_h_e=self.ent_embeddings(pos_h).view(-1,self.config.ent_size,1)
# 		p_t_e=self.ent_embeddings(pos_t).view(-1,self.config.ent_size,1)
# 		p_r_e=self.rel_embeddings(pos_r).view(-1,self.config.rel_size)
# 		n_h_e=self.ent_embeddings(neg_h).view(-1,self.config.ent_size,1)
# 		n_t_e=self.ent_embeddings(neg_t).view(-1,self.config.ent_size,1)
# 		n_r_e=self.rel_embeddings(neg_r).view(-1,self.config.rel_size)
# 		p_matrix=self.transfer_matrix(pos_r).view(-1,self.config.rel_size,self.config.ent_size)
# 		n_matrix=self.transfer_matrix(neg_r).view(-1,self.config.rel_size,self.config.ent_size)
# 		p_h=self._transfer(p_matrix,p_h_e).view(-1,self.config.rel_size)
# 		p_t=self._transfer(p_matrix,p_t_e).view(-1,self.config.rel_size)
# 		p_r=p_r_e
# 		n_h=self._transfer(n_matrix,n_h_e).view(-1,self.config.rel_size)
# 		n_t=self._transfer(n_matrix,n_t_e).view(-1,self.config.rel_size)
# 		n_r=n_r_e
# 		_p_score=self._calc(p_h, p_t, p_r)
# 		_n_score=self._calc(n_h, n_t, n_r)
# 		_p_score=_p_score.view(-1, 1, self.config.entity_embedding_dimension)
# 		_n_score=_n_score.view(-1, self.config.negative_ent + self.config.negative_rel, self.config.entity_embedding_dimension)
# 		p_score=torch.sum(torch.mean(_p_score, 1),1)
# 		n_score=torch.sum(torch.mean(_n_score, 1),1)
# 		loss=self.loss_func(p_score,n_score)
# 		return loss
# 	def predict(self, predict_h, predict_t, predict_r):
# 		p_h_e=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda()).view(-1,self.config.ent_size,1)
# 		p_t_e=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda()).view(-1,self.config.ent_size,1)
# 		p_r_e=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda()).view(-1,self.config.rel_size)
# 		p_matrix=self.transfer_matrix(Variable(torch.from_numpy(predict_r)).cuda()).view(-1,self.config.rel_size,self.config.ent_size)
# 		p_h=self._transfer(p_matrix,p_h_e).view(-1,self.config.rel_size)
# 		p_t=self._transfer(p_matrix,p_t_e).view(-1,self.config.rel_size)
# 		p_r=p_r_e
# 		_p_score = self._calc(p_h, p_t, p_r)
# 		p_score=torch.sum(_p_score,1)
# 		return p_score.cpu()

