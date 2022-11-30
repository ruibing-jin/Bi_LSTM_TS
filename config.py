
import yaml
import os
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.gpu = '0'
config.save_frequency = 1
config.seed = 1

#distributed training
config.dist_backend = 'nccl'
config.world_size = -1
config.rank = -1
config.dist_url = 'tcp://224.66.41.62:23456'
config.multiprocessing_distributed = False
config.distributed = False
config.task = ''

#dataset
config.data = edict()
config.data.root =''
config.data.set = ''
config.data.max_rul = 130
config.data.seq_len = 15
config.data.num_worker = 4
config.data.input_type = ''
config.data.test_id = 0

#network
config.net = edict()
config.net.name = 'bilstm'
config.net.hand_craft = False

config.net.num_hidden = 18
config.net.input_dim = 9
config.net.aux_dim = 4
config.net.hand_dim = 0


#train
config.train = edict()
config.train.resume_epoch = False
config.train.fine_tune = True
config.train.batch_size = 4
config.train.lr = 0.01
config.train.lr_factor = 0.1
config.train.end_epoch = 2
config.train.callback_freq = 50
config.train.optimizer = 'sgd'
config.train.warmup_iters = 0
config.train.lr_mult = 0.2

#test
config.test = edict()
config.test.model_name = ''
config.test.model_path = ''

def update_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(config_file)
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vv == 'None':
                            config[k][vk] = None
                        else:
                            config[k][vk] = vv
                else:
                    if v == 'None':
                        config[k] = None
                    elif k == 'lr_epoch':
                        step_list = [int(x) for x in v.split(',')]
                        config[k] = step_list
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py: {:}".format(k))