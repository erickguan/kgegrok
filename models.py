import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torch.autograd import Variable

def _get_all_instance(batch, negative_batch):
    """returns the tuple of all h, r, t including negative samples.
    """

    return

class Model(nn.Module):
	def __init__(self, config):
		super(Model, self).__init__()
		self.config = config

	def get_postive_instance(self):
		self.postive_h = Variable(torch.from_numpy(self.config.batch_h[0:self.config.batch_size])).cuda()
		self.postive_t = Variable(torch.from_numpy(self.config.batch_t[0:self.config.batch_size])).cuda()
		self.postive_r = Variable(torch.from_numpy(self.config.batch_r[0:self.config.batch_size])).cuda()
		return self.postive_h,self.postive_t,self.postive_r

	def get_negtive_instance(self):
		self.negtive_h = Variable(torch.from_numpy(self.config.batch_h[self.config.batch_size:self.config.batch_seq_size])).cuda()
		self.negtive_t = Variable(torch.from_numpy(self.config.batch_t[self.config.batch_size:self.config.batch_seq_size])).cuda()
		self.negtive_r = Variable(torch.from_numpy(self.config.batch_r[self.config.batch_size:self.config.batch_seq_size])).cuda()
		return self.negtive_h,self.negtive_t,self.negtive_r

	def get_all_instance(self):
		self.batch_h = Variable(torch.from_numpy(self.config.batch_h)).cuda()
		self.batch_t = Variable(torch.from_numpy(self.config.batch_t)).cuda()
		self.batch_r = Variable(torch.from_numpy(self.config.batch_r)).cuda()
		return self.batch_h, self.batch_t, self.batch_r

	def get_all_labels(self):
		self.batch_y=Variable(torch.from_numpy(self.config.batch_y)).cuda()
		return self.batch_y

	def predict(self):
		pass

	def forward(self, batch, negative_batch):
		raise NotImplementedError

	def loss_func(self):
		pass


class TransE(Model):
    '''TransE is the first model to introduce translation-based embedding,
    which interprets relations as the translations operating on entities.
    '''
	def __init__(self, config):
		super(TransE, self).__init__(config)
		self.ent_embeddings = nn.Embedding(config.entTotal,config.hidden_size)
		self.rel_embeddings = nn.Embedding(config.relTotal,config.hidden_size)
		self.init_weights()

	def init_weights(self):
		nn.init.xavier_uniform(self.ent_embeddings.weight.data)
		nn.init.xavier_uniform(self.rel_embeddings.weight.data)

	def _calc(self,h,t,r):
		return torch.abs(h + r - t)

	# margin-based loss
	def loss_func(self,p_score,n_score):
		criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
		y = Variable(torch.Tensor([-1])).cuda()
		loss = criterion(p_score,n_score,y)
		return loss

	def forward(self):
		pos_h,pos_t,pos_r=self.get_postive_instance()
		neg_h,neg_t,neg_r=self.get_negtive_instance()
		p_h=self.ent_embeddings(pos_h)
		p_t=self.ent_embeddings(pos_t)
		p_r=self.rel_embeddings(pos_r)
		n_h=self.ent_embeddings(neg_h)
		n_t=self.ent_embeddings(neg_t)
		n_r=self.rel_embeddings(neg_r)
		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)
		_p_score = _p_score.view(-1, 1, self.config.hidden_size)
		_n_score = _n_score.view(-1, self.config.negative_ent + self.config.negative_rel, self.config.hidden_size)
		p_score=torch.sum(torch.mean(_p_score, 1),1)
		n_score=torch.sum(torch.mean(_n_score, 1),1)
		loss=self.loss_func(p_score, n_score)
		return loss

	def predict(self, predict_h, predict_t, predict_r):
		p_h=self.ent_embeddings(Variable(torch.from_numpy(predict_h)).cuda())
		p_t=self.ent_embeddings(Variable(torch.from_numpy(predict_t)).cuda())
		p_r=self.rel_embeddings(Variable(torch.from_numpy(predict_r)).cuda())
		_p_score = self._calc(p_h, p_t, p_r)
		p_score=torch.sum(_p_score,1)
		return p_score.cpu()
