#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-24 16:41:03
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

from solve_data import delete_features
from save_load_result import load_result, save_result
from map_features_to_digit import convert_to_numerical

from sklearn.cross_validation import StratifiedKFold
from Module_roc import calculate_draw_roc
import numpy as np
from sklearn.metrics import roc_curve, auc

import xgboost as xgb


'''
xgboost.DMatrix(data, label=None, missing=None, 
				weight=None, silent=False, feature_names=None, feature_types=None)

# Train a booster with given parameters.
xgboost.train(params, dtrain, num_boost_round=10, evals=(), obj=None, 
				feval=None, maximize=False, early_stopping_rounds=None, 
				evals_result=None, verbose_eval=True, learning_rates=None, xgb_model=None)

# Cross-validation with given paramaters.
 xgboost.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, 
 				folds=None, metrics=(), obj=None, feval=None, maximize=False, 
 				early_stopping_rounds=None, fpreproc=None, as_pandas=True, 
 				verbose_eval=None, show_stdv=True, seed=0)

#  predict(data, output_margin=False, ntree_limit=0, pred_leaf=False)

    Predict with data.
'''

def use_cv_to_choose_best_params(xgtrain, label, test_params):
	paramslst = list(params.items())
	cv_res = xgb.cv(paramslst, data = train, nfold = 5, label = label)

	#calculate_draw_roc()

def module_xgboost_pre(train, labels, test):

	'''
	Parameters for Tree Booster:
	eta: step size shrinkage used in update to prevents overfitting [0,1]
		can also reduce stepsize eta, but needs to remember to increase num_round when you do so.

	##### control model complexity ####
	gamma: minimum loss reduction required to make a further partition on a leaf node of the tree [0,∞]
	max_depth: maximum depth of a tree [1,∞]
	min_child_weight: minimum sum of instance weight(hessian) needed in a child  [0,∞]
	max_delta_step: Maximum delta step we allow each tree’s weight estimation to be[0,∞]
					it might help in logistic regression when class is extremely imbalanced
					Set it to value of 1-10 might help control the update

	##### add randomness to make training robust to noise
	subsample: subsample ratio of the training instance (0,1]
	colsample_bytree: subsample ratio of columns when constructing each tree (0,1]
	'''


	params = {} # Booster params.
	params["objective"] = "binary:logistic" # Specify the learning task and the corresponding learning objective
	params["eta"] = 0.4 # step size shrinkage used in update to prevents overfitting

	params["min_child_weight"] = 6 # minimum sum of instance weight(hessian) needed in a child
	params["max_depth"] = 6
	params["max_delta_step"] = 1

	params["subsample"] = 0.5 #subsample ratio of the training instance
	params["colsample_bytree"] = 0.7 # subsample ratio of columns when constructing each tree
	# ???????????? scale_pos_weight ?????? how to use
	#params["scale_pos_weight"] = 1 # Balance the positive and negative weights, via scale_pos_weight
	params["eval_metric"] = "auc" # this must be auc

	plst = list(params.items())

	num_rounds = 10000

	#Using 1/5 of of train rows for early stopping. 

	xgtest = xgb.DMatrix(test)

	offset = int(len(train) / 5)
	#create a train and validation dmatrices 
	xgtrain = xgb.DMatrix(train[offset:, :], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset, :], label=labels[:offset])

	#train using early stopping and predict
	watchlist = [(xgtrain, 'train'),(xgval, 'eval')]

	model = xgb.train(plst, xgtrain, num_boost_round = num_rounds, evals = watchlist, early_stopping_rounds=120)
	preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

	model.save_model('0001.model')

	#calculate_draw_roc()


	#reverse train and labels and use different 5k for early stopping. 
	# this adds very little to the score but it is an option if you are concerned about using all the data. 
	train = train[::-1,:]
	labels = np.log(labels[::-1])

	xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
	xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

	watchlist = [(xgtrain, 'train'),(xgval, 'eval')]
	model = xgb.train(plst, xgtrain, num_boost_round = num_rounds, evals = watchlist, early_stopping_rounds=120)
	preds2 = model.predict(xgtest, ntree_limit=model.best_iteration)

	model.save_model('0001_1.model')
	#combine predictions
	#since the metric only cares about relative rank we don't need to average
	preds = (preds1)*1.4 + (preds2)*8.6
	return preds



if __name__ == '__main__':
	contents = load_result("data_after_features_processed.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	label_lines = np.array(load_result("train_label_original.csv"))
	#print(label_lines.shape)
	from save_load_result import convert_to_int
	label = convert_to_int(label_lines)

	label = label.reshape((label.size, ))
	print(label.shape)

	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	data = convert_to_numerical(data, features)

	test_data = data[:2000]
	test_label = label[:2000]
	train_data = data[2000:]
	train_label = label[2000:]

	test_preds = module_xgboost_pre(train_data, train_label, test_data)
	calculate_draw_roc(test_label, test_preds, save_fig_name = "module_xgb_ROC.png")