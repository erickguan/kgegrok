import logging
import numpy as np
import sys
import select
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp
from sklearn.model_selection import ParameterGrid
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from kgegrok import data
from kgegrok import estimate
from kgegrok import stats
from kgegrok import evaluation
from kgegrok import utils
from kgegrok.stats import create_drawer


bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='/MIUN/.bert')
bert.cuda()
bert.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/MIUN/.bert')

# # Tokenized input
# text = "Who was Jim Henson ? Jim Henson was a puppeteer"
# tokenized_text = tokenizer.tokenize(text)

# # Mask a token that we will try to predict back with `BertForMaskedLM`
# masked_index = 6
# tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

# # Convert token to vocabulary indices
# indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
# segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# # Convert inputs to PyTorch tensors
# tokens_tensor = torch.tensor([indexed_tokens])
# segments_tensors = torch.tensor([segments_ids])

# encoded_layers, a = model(tokens_tensor.cuda(), segments_tensors.cuda())

# print(encoded_layers[-1].shape, a.shape)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    utils.report_gpu_info()

    default_args = {
        'data_dir': 'data/FB15k-237',
        'triple_order': 'htr',
        'negative_entity': 1,
        'entity_embedding_dimension': 50,
        'alpha': 0.1,
        'delimiter': ' ',
        'batch_size': 10000,
        'num_workers': 10,
        'num_evaluation_workers': 10,
        'model': "ComplExText",
        'optimizer': "SGD",
        'margin': 1.0,
        'epochs': 100,
        'name': 'test',
        'plot_graph': False,
        'lambda_': 0.001,
        'enable_cuda': False,
        'evaluation_load_factor': 0.05
    }
    config = utils.build_config_with_dict(default_args)
    print(config.__dict__)

    triple_source = data.TripleSource(config.data_dir, config.triple_order,
                                    config.triple_delimiter)
    model_class = utils.load_class_from_module(config.model, 'kgegrok.models',
                                            'kgegrok.text_models')

    with evaluation.validation_resource_manager('train', config, triple_source) as pool:
        utils.seed_modules(
            config,
            numpy_seed=10000,
            torch_seed=20000,
            torcu_cuda_seed_all=2192,
            cuda_deterministic=True,
            kgegrok_base_seed=30000,
            cuda_benchmark=config.cudnn_benchmark)

        optimizer_class = utils.load_class_from_module(config.optimizer,
                                                        'torch.optim')
        model = model_class(triple_source, config)
        if config.enable_cuda:
            model = nn.DataParallel(model)
            # has to be here because https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/7
            model.cuda()
        optimizer = create_optimizer(optimizer_class, config, model.parameters())

        for i_epoch in range(INDEX_OFFSET, config.epochs + INDEX_OFFSET, 1):
            model.train()
            logging.info('--------------------')
            logging.info('Training at epoch ' + str(i_epoch))
            logging.info('--------------------')

            loss_epoch = 0.0
            i_batch = 0
            for sample_batched in data_loader:
                logging.info('Training batch ' + str(i_batch + INDEX_OFFSET) + "/" +
                            str(len(data_loader.dataset)))

                loss = model.forward(sample_batched)
                loss_sum = loss.sum()
                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                loss_epoch += float(loss_sum.item()) # avoid long-term memory usage
                i_batch += 1
