
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

####################
# COMMON METRICS
####################

class EvalMetric(object):

	def __init__(self, name, **kwargs):
		self.name = str(name)
		self.reset()

	def update(self, preds, labels, losses):
		raise NotImplementedError()

	def reset(self):
		self.num_inst = 0
		self.sum_metric = 0.0

	def get(self):
		if self.num_inst == 0:
			return (self.name, float('nan'))
		else:
			return (self.name, self.sum_metric / self.num_inst)

	def get_name_value(self):
		name, value = self.get()
		if not isinstance(name, list):
			name = [name]
		if not isinstance(value, list):
			value = [value]
		return list(zip(name, value))

	def check_label_shapes(self, preds, labels):
		# raise if the shape is inconsistent
		if (type(labels) is list) and (type(preds) is list):
			label_shape, pred_shape = len(labels), len(preds)
		else:
			label_shape, pred_shape = labels.shape[0], preds.shape[0]

		if label_shape != pred_shape:
			raise NotImplementedError("")


class MetricList(EvalMetric):
	"""Handle multiple evaluation metric
	"""
	def __init__(self, *args, name="metric_list"):
		assert all([issubclass(type(x), EvalMetric) for x in args]), \
			"MetricList input is illegal: {}".format(args)
		self.metrics = [metric for metric in args]
		super(MetricList, self).__init__(name=name)

	def update(self, preds, labels, losses=None):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels
		losses = [losses] if type(losses) is not list else losses

		for metric in self.metrics:
			metric.update(preds, labels, losses)

	def reset(self):
		if hasattr(self, 'metrics'):
			for metric in self.metrics:
				metric.reset()
		else:
			logging.warning("No metric defined.")

	def get(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get())
		return ouputs

	def get_name_value(self):
		ouputs = []
		for metric in self.metrics:
			ouputs.append(metric.get_name_value())        
		return ouputs

####################
# RUL EVAL METRICS
####################

class RMSE(EvalMetric):
      
	def __init__(self, max_rul, name='RMSE'):
		super(RMSE, self).__init__(name)
		self.max_rul = max_rul

	def update(self, preds, labels, losses):
		assert losses is not None, "Loss undefined."

		for loss in losses:
			self.sum_metric += float(self.max_rul * self.max_rul* loss.numpy().sum())*labels[0].shape[0]
			self.num_inst += labels[0].shape[0]
	
	def get(self):
		if self.num_inst == 0:
			return (self.name, float('nan'))
		else:
			return (self.name, np.sqrt(self.sum_metric / self.num_inst))
	
class RULscore(EvalMetric):
	"""Computes RUL score.
	"""
	def __init__(self, max_rul, name='RULscore'):
		super(RULscore, self).__init__(name)
		self.max_rul = max_rul

	def update(self, preds, labels, losses):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels

		self.check_label_shapes(preds, labels)
		for pred, label in zip(preds[0], labels[0]):
			if label <= pred:
				diff_val = (-1 * self.max_rul * (label.numpy() - pred.numpy())/10.0)[0]
				if diff_val >= 14: diff_val = 14
				self.sum_metric = self.sum_metric + np.exp(diff_val)-1
			else:
				diff_val = (self.max_rul * (label.numpy() - pred.numpy())/13.0)[0]
				if diff_val >= 14: diff_val = 14
				self.sum_metric = self.sum_metric + np.exp(diff_val)-1
	
	def get(self):

		return (self.name, self.sum_metric)

class meanRULscore(EvalMetric):
	"""Computes RUL score.
	"""
	def __init__(self, max_rul, name='meanRULscore'):
		super(meanRULscore, self).__init__(name)
		self.max_rul = max_rul

	def update(self, preds, labels, losses):
		preds = [preds] if type(preds) is not list else preds
		labels = [labels] if type(labels) is not list else labels

		self.check_label_shapes(preds, labels)
		for pred, label in zip(preds[0], labels[0]):
			if label <= pred:
				self.sum_metric = self.sum_metric + np.exp(-1 * self.max_rul * (label.numpy() - pred.numpy())/10.0)[0]-1
			else:
				self.sum_metric = self.sum_metric + np.exp(self.max_rul * (label.numpy() - pred.numpy())/13.0)[0]-1
			self.num_inst += 1.0
