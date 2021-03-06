import os
import logging
import random
import numpy as np
import torch


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def freeze_layer(layer):
    for param in layer.parameters():
        param.requires_grad = False


def adjust_learning_rate(optimizer, lr):
    """
    checked
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save(filepath, filename, model, optimizer, epoch, best_metric):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    model_path = '%s/%s' % (filepath, filename)
    logging.info("saving model: %s" % model_path)
    torch.save({'model': model, 'optimizer': optimizer, 'epoch': epoch, 'best_metric': best_metric}, model_path)


def load(filepath):
    logging.info('loading model: %s' % filepath)
    ckpt = torch.load(filepath)
    return ckpt['model'], ckpt['optimizer'], ckpt['epoch'], ckpt['best_metric']
