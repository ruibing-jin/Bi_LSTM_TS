
import os
import argparse
import pprint
from datetime import date
import _init_paths

import numpy as np
import random
import warnings
import torch


from config import config, update_config
import create_logger
from lib import metric
from core.model import model
from dataloader import data_loader as data_loader
from networks.two_stream_bilstm import Bi_LSTM


torch.backends.cudnn.enabled = False

def parse_args():
    parser = argparse.ArgumentParser(description='Code for time series RUL by Ruibing')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()

    # update config
    update_config(args.cfg)

    return args

if __name__ == "__main__":

	# set args
	args = parse_args()
	curr_path = os.path.abspath(os.path.dirname(__file__))
	logger, final_output_path, model_fixtime = create_logger.create_logger(curr_path, args.cfg, config)
	config.update({'output_pt': final_output_path})
	logger.info('training config:{}\n'.format(pprint.pformat(config)))

	os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

	if config.seed:
		random.seed(config.seed)
		np.random.seed(config.seed)
		torch.manual_seed(config.seed)
		torch.cuda.manual_seed(config.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		warnings.warn('You have chosen to seed training. '
					  'This will turn on the CUDNN deterministic setting, '
					  'which can slow down your training considerably! '
					  'You may see unexpected behavior when restarting '
					  'from checkpoints.')

	data_iter = data_loader.CMPDataIter(config.data.root, config.data.set, config.data.max_rul, config.data.seq_len)

	alldata_loader = torch.utils.data.DataLoader(data_iter, batch_size=config.train.batch_size, num_workers=config.data.num_worker,
												pin_memory=True, worker_init_fn=data_loader.worker_init_fn)
												
	sym_net = Bi_LSTM(num_hidden = config.net.num_hidden, input_dim = config.net.input_dim, aux_dim = config.net.aux_dim, hand_dim = config.net.hand_dim)

	model_prefix = os.path.join(config.output_pt, 'exp_'+ model_fixtime + '_' + config.net.name)
	net = model(net=sym_net, criterion=torch.nn.MSELoss().cuda(), model_prefix=model_prefix, step_callback_freq=config.train.callback_freq,
				save_checkpoint_freq=config.save_frequency, logger = logger)
	net.net.cuda()
	net.net = torch.nn.DataParallel(net.net).cuda()

	if config.train.optimizer.lower() == 'sgd':
		optimizer = torch.optim.SGD(net.net.parameters(), 
									lr=config.train.lr, 
									momentum=0.9, 
									weight_decay=0.0001, 
									nesterov=True)
	elif config.train.optimizer.lower() == 'adam':
		optimizer = torch.optim.Adam(net.net.parameters(),
									lr=config.train.lr,
									weight_decay=0.0001)
	elif config.train.optimizer.lower() == 'adamw':
		optimizer = torch.optim.AdamW(net.net.parameters(),
									lr=config.train.lr,
									weight_decay=0.0001)
	elif config.train.optimizer.lower() == 'rmsprop':
		optimizer = torch.optim.RMSprop(net.net.parameters(),
									lr=config.train.lr,
									weight_decay=0.0001)
	else:
		raise NotImplementedError(config.train.optimizer.lower())
	

	lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = \
					[int(x) for x in config.train.lr_epoch], gamma=config.train.lr_factor)

	metrics = metric.MetricList(metric.RMSE(max_rul = config.data.max_rul), metric.RULscore(max_rul = config.data.max_rul),)

	net.fit(data_iter=alldata_loader, dataset = data_iter, optimizer=optimizer, lr_scheduler=lr_scheduler, metrics=metrics,
			epoch_start=0, epoch_end=config.train.end_epoch)