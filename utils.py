import torch
import os.path
import json
from pathlib import Path


def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))

def save_checkpoint(state, filename='model_states/checkpoint.pth.tar', postfix_num=None):
    path = "{}_{}".format(filename, postfix_num) if postfix_num is not None else filename
    dirname = os.path.dirname(path)
    Path(dirname).mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(model, optimizer, config):
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


class _VisdomWindowDataReader(object):
    def __init__(self, drawer):
        self.drawer = drawer

    def __call__(self, win):
        content = self.drawer.get_window_data(win)
        if content is None or len(content) == 0:
            content = "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(content)

def write_logging_data(drawer, windows, config):
    """writes the logging data."""
    if config.logging_path is None or config.name is None:
        return
    r = _VisdomWindowDataReader(drawer)
    result = list(map(r, windows))
    with open(os.path.join(config.logging_path, config.name), 'w') as f:
        f.write(json.dumps(result))
