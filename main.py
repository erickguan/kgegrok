import torch
import data
import kgekit
import models
from estimate import train_and_validate
import logging
import os
import torch.optim as optim
import visdom


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
    test_head = True
    test_relation = False
    test_tail = True
    alpha = 0.001
    report_features = data.LinkPredictionStatistics.DEFAULT
    report_dimension = data.StatisticsDimension.DEFAULT

def cli():
    config = Config()
    train_and_validate(config, models.TransE, optim.Adam, visdom.Visdom(port=6006))

def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    report_gpu_info()

    cli()

