import torch
import data
import kgekit
import models
from estimate import train_and_validate
import logging
import torch.optim as optim
import visdom
import numpy as np

class Config(object):
    data_dir = "data/YAGO3-10"
    triple_order = "hrt"
    triple_delimiter = ' '
    negative_entity = 5
    negative_relation = 1
    batch_size = 100
    num_workers = 2
    entity_embedding_dimension = 50
    margin = 1.0
    epoches = 1
    alpha = 0.001
    lambda_ = 0.001
    report_features = data.LinkPredictionStatistics.DEFAULT
    report_dimension = data.StatisticsDimension.DEFAULT
    # filename to resume
    resume = None
    # Introduce underministic behaviour but allow cudnn find best algoritm
    cudnn_benchmark = True

def cli():
    config = Config()
    np.random.seed(10000)
    torch.manual_seed(20000)
    torch.cuda.manual_seed_all(2192)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    model = train_and_validate(config, models.TransE, optim.Adam, visdom.Visdom(port=6006))

def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report_gpu_info()

    cli()

