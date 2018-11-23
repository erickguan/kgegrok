import torch
import os.path
import json
from pathlib import Path
import importlib
import kgekit.io


def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))

def save_checkpoint(state, filename='model_states/checkpoint.pth.tar', postfix_num=None):
    path = "{}_{}".format(filename, postfix_num) if postfix_num is not None else filename
    dirname = os.path.dirname(path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(config, model, optimizer=None):
    if config.resume:
        if os.path.isfile(config.resume):
            logging.info("loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info("loaded checkpoint '{}' (epoch {})"
                  .format(config.resume, checkpoint['epoch']))
        else:
            logging.info("no checkpoint found at '{}'".format(config.resume))

def write_logging_data(raw_data, config):
    """writes the logging data."""
    with open(os.path.join(config.logging_path, config.name), 'w') as f:
        f.write(json.dumps(raw_data))

def load_class_from_module(class_name, *modules):
    for module in modules:
        mod = importlib.import_module(module)
        try:
            return getattr(mod, class_name)
        except:
            pass
    raise RuntimeError("Can't find the {} from {}".format(class_name, modules))

def read_triple_translation(config):
    translation_path = os.path.join(config.data_dir, config.translation_filename)
    entities, relations = kgekit.io.read_translation(translation_path)
    return dict(entities), dict(relations)
