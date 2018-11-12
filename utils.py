import torch
import os.path
import json


def report_gpu_info():
    count = torch.cuda.device_count()
    for i in range(count):
        print(str(i) + " " + torch.cuda.get_device_name(i))

def save_checkpoint(state, filename='model_states/checkpoint.pth.tar', postfix_num=None):
    path = "{}_{}".format(filename, postfix_num) if postfix_num is not None else filename
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


def write_logging_data(drawer, windows, config):
    """writes the logging data."""
    if config.logging_path is None or config.name is None:
        return
    result = list(map(lambda win: json.loads(drawer.get_window_data(win), windows)))
    with open(os.path.join(config.logging_path, config.name), 'w') as f:
        f.write(json.dumps(result))
