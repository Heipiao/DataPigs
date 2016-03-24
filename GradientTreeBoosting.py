#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-14 20:43:41
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: GradientBoostingClassifier supports both binary and multi-class classification

from solve_data import delete_features
from save_load_result import load_result, save_result
from map_features_to_digit import convert_to_numerical

from sklearn.cross_validation import StratifiedKFold


import numpy as np

'''
class sklearn.ensemble.GradientBoostingClassifier(loss='deviance', learning_rate=0.1, \
				n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, \
				min_weight_fraction_leaf=0.0, max_depth=3, init=None, \
				random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, \
				warm_start=False, presort='auto')

shrinkage and subsampling should be used together
A typical value of subsample is 0.5
learning_rate <= 0.1
'''
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy import interp



def use_xgboost():
	pass


def prepare_data_for_using(data, label):
	pass

def use_GRClassifier():
	original_params = {"loss": 'exponential', "n_estimators": 1000, "max_leaf_nodes": 4, "max_depth": None, \
						"min_samples_split": 5}

	setting = {"learning_rate": 0.05, "subsample": 0.5}
	params = dict(original_params)
	params.update(setting)

	clf = GradientBoostingClassifier(**params)
	return clf


# cv_Flod = StratifiedKFold(y, n_folds=6) 
def calculate_draw_roc(classifier, data, features, label, cv_Flod, original_data, original_label):
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	my_test = original_data[:400]
	my_label = original_label[:400]

	features_importance = dict()

	for i, (train, test) in enumerate(cv_Flod):
	    
	    fitted_classifier = classifier.fit(data[train], label[train])
	    probas_ = fitted_classifier.predict_proba(data[test])
	    if i == 1:
	    	save_result(probas_, "predict_result.csv")
	    	save_result(label[test], "original_result.csv")

	    # Compute ROC curve and area the curve
	    fpr, tpr, thresholds = roc_curve(label[test], probas_[:, 1])
	    mean_tpr += interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    roc_auc = auc(fpr, tpr)
	    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))


	    importances = fitted_classifier.feature_importances_

	    indices = np.argsort(importances)[::-1]

	    print("Feature ranking: ")
	    for f in range(data.shape[1]):
	    	print("%s. %d (%f)" % (features[indices[f]], indices[f], importances[indices[f]]))
	    	features_importance[features[indices[f]]] = importances[indices[f]]

	test_probs = fitted_classifier.predict_proba(my_test)
	test_fpr, test_tpr, test_thresholds = roc_curve(my_label, test_probs[:, 1])
	roc_auc = auc(test_fpr, test_tpr)
	plt.plot(test_fpr, test_tpr, lw=1, label='ROC test (area = %0.2f)' % (roc_auc))



	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(cv_Flod)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--',
	         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig("ROC_GB_user_all_solved_lr(0.05).png")

	return features_importance


if __name__ == '__main__':
	contents = load_result("data_after_features_processed.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	label_lines = np.array(load_result("train_label_original.csv"))
	#print(label_lines.shape)
	from save_load_result import convert_to_float
	label = convert_to_float(label_lines)

	label = label.reshape((label.size, ))

	# from create_new_features import find_featuers_index
	# features_name = "WeblogInfo"
	# fea_indexs = find_featuers_index(features_name, features)
	# print(fea_indexs)

	# data = data[:, fea_indexs]
	# features = features[fea_indexs]

	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	data = convert_to_numerical(data, features)

	classifier = use_GRClassifier()
	cv_Flod = StratifiedKFold(label[1000:], n_folds=3) 
	calculate_draw_roc(classifier, data[1000:], features, label[1000:], cv_Flod, data, label)

