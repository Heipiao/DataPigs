#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-05 15:30:52
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import numpy as np
import matplotlib.pyplot as plt

from save_load_result import load_result, convert_to_float
from draw_result import plot_pr
from map_features_to_digit import convert_to_numerical

from sklearn.cross_validation import train_test_split  
from sklearn.metrics import precision_recall_curve, roc_curve, auc  
from sklearn.metrics import classification_report  
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold
from scipy import interp
from save_load_result import save_result

from features_reduce import use_RandomForestRegressor_to_delete

def use_LR_to_delete(data, features, label):
	from sklearn.linear_model import LinearRegression
	lr = LinearRegression()
	lr.fit(data, label)
	#A helper method for pretty-printing linear models
	def pretty_print_linear(coefs, names = None, sort = False):
	  if names == None:
	    names = ["X%s" % x for x in range(len(coefs))]
	  lst = zip(coefs, names)
	  if sort:
	    lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
	  return " + ".join("%s * %s" % (round(coef, 3), name)
	                   for coef, name in lst)
	print("Linear model:", pretty_print_linear(lr.coef_, features[1:]))


if __name__ == '__main__':

	contents = load_result("after_delete_strong_correlation_features_data.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])
	label_lines = np.array(load_result("train_label_original.csv"))
	data = convert_to_numerical(data, features)

	label = convert_to_float(label_lines)
	label = label.reshape((label.size, ))

	#use_RandomForestRegressor_to_delete(data, features, label)
	use_LR_to_delete(data, features, label)
#################### first example #######################
	# testNum = 10
	# average = 0
	# for i in range(0, testNum):  
	#     #加载数据集，切分数据集80%训练，20%测试  
	#     x_train, x_test, y_train, y_test \
	#         = train_test_split(data, label, test_size = 0.2)  
	  
	#     #训练LR分类器  
	#     clf = LogisticRegression()  
	#     clf.fit(x_train, y_train)  
	#     y_pred = clf.predict(x_test)  
	#     p = np.mean(y_pred == y_test)  
	#     print(p)  
	#     average += p  
  
      
	# #准确率与召回率  
	# answer = clf.predict_proba(x_test)[:,1]  
	# precision, recall, thresholds = precision_recall_curve(y_test, answer)      
	# report = answer > 0.5  
	# print(classification_report(y_test, report, target_names = ['neg', 'pos']))  
	# print("average precision:", average/testNum)    
	  
	# plot_pr(0.5, precision, recall, "pos")  




	# cv = StratifiedKFold(label, n_folds=5)
	# classifier = LogisticRegression()

	# mean_tpr = 0.0
	# mean_fpr = np.linspace(0, 1, 100)
	# all_tpr = []

	# for i, (train, test) in enumerate(cv):
	#     probas_ = classifier.fit(data[train], label[train]).predict_proba(data[test])
	#     # Compute ROC curve and area the curve
	#     #print(classifier.get_params())
	#     fpr, tpr, thresholds = roc_curve(label[test], probas_[:, 1])
	#     mean_tpr += interp(mean_fpr, fpr, tpr)
	#     mean_tpr[0] = 0.0
	#     roc_auc = auc(fpr, tpr)
	#     plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

	# plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	# mean_tpr /= len(cv)
	# mean_tpr[-1] = 1.0
	# mean_auc = auc(mean_fpr, mean_tpr)
	# plt.plot(mean_fpr, mean_tpr, 'k--',
	#          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	# plt.xlim([-0.05, 1.05])
	# plt.ylim([-0.05, 1.05])
	# plt.xlabel('False Positive Rate')
	# plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic example')
	# plt.legend(loc="lower right")
	# plt.savefig("test_method2.png")
	# plt.show()



	#A helper method for pretty-printing linear models
	# def pretty_print_linear(coefs, names = None, sort = False):
	#   if names == None:
	#     names = ["X%s" % x for x in range(len(coefs))]
	#   lst = zip(coefs, names)
	#   if sort:
	#     lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
	#   return " + ".join("%s * %s" % (round(coef, 3), name)
	#                    for coef, name in lst)
	# print("Linear model:", pretty_print_linear(classifier.coef_))