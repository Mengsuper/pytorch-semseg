import copy
import logging
import functools

from ptsemseg.loss.loss import cross_entropy2d
from ptsemseg.loss.loss import bootstrapped_cross_entropy2d
from ptsemseg.loss.loss import multi_scale_cross_entropy2d
from ptsemseg.loss.loss import chrom_downsampling_loss
from ptsemseg.loss.loss import ssim_loss_function


logger = logging.getLogger('ptsemseg')

key2loss = {'chrom_downsampling_loss': chrom_downsampling_loss,
            'cross_entropy': cross_entropy2d,
            'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
            'multi_scale_cross_entropy': multi_scale_cross_entropy2d,
            'ssim_loss_function': ssim_loss_function}

def get_loss_function(cfg):
    if cfg['training']['loss'] is None:
        logger.info("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        logger.info('Using {} with {} params'.format(loss_name, 
                                                     loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
