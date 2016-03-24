#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-24 18:18:37
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
# test_probs = fitted_classifier.predict_proba(my_test)

def calculate_draw_roc(my_label, test_probs, save_fig_name = "ROC.png"):

	test_fpr, test_tpr, test_thresholds = roc_curve(my_label, test_probs[:, 1])
	roc_auc = auc(test_fpr, test_tpr)
	plt.plot(test_fpr, test_tpr, lw=1, label='ROC test (area = %0.2f)' % (roc_auc))

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	# mean_tpr /= len(cv_Flod)
	# mean_tpr[-1] = 1.0
	# mean_auc = auc(mean_fpr, mean_tpr)
	# plt.plot(mean_fpr, mean_tpr, 'k--',
	#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig(save_fig_name)

	return roc_auc