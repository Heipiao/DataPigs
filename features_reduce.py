#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-03 08:44:59
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

from save_load_result import load_result, save_result
from solve_data import get_known_features_index
from map_features_to_digit import convert_to_numerical
import scipy.stats as stats
import numpy as np
import csv

# use the pearsonr --> to calculate correlation between Numeric style 
# use the spearmanr --> to calculate correlation between str and other style 
# output:
#	all the compared two properties whose correlation are bigger than 0.7
# range:
# 相关系数:
	# 0.8-1.0 极强相关
	# 0.6-0.8 强相关
	# 0.4-0.6 中等程度相关
	# 0.2-0.4 弱相关
	# 0.0-0.2 极弱相关或无相关
def correlation_between_properties(data, features):
	fixed_str_features = np.array(load_result("str_features.csv"))[0]
	indexs = get_known_features_index(features, fixed_str_features)

	title = list()
	title.append("features1")
	title.append("features2")
	title.append("calculate_method") 
	title.append("cor")
	title.append("pval")
	save_result(title, "pearsonr_spearmanr_results.csv")
	save_result(title, "pearsonr_spearmanr_Strong_correlation.csv")
	for fea_pos in range(len(features)):
		for fea_pos_add in range(fea_pos + 1, len(features)):
			info_result = list()
			info_result.append(features[fea_pos])
			info_result.append(features[fea_pos_add])
			a1 = data[:, fea_pos]
			a2 = data[:, fea_pos_add]
			# they are all not str style features
			if fea_pos not in indexs and fea_pos_add not in indexs:
				info_result.append("pearsonr")
				cor, pval = stats.pearsonr(a1, a2)
			else: # one of them or all of them are str style features
				info_result.append("spearmanr")
				cor, pval = stats.spearmanr(a1, a2)
			cor = round(cor, 3)
			info_result.append(cor)
			info_result.append(pval)
			if abs(cor) >= 0.2:
				save_result(info_result, "pearsonr_spearmanr_results.csv", style = "a+")
			if abs(cor) >= 0.86:
				save_result(info_result, "pearsonr_spearmanr_Strong_correlation.csv", \
												style = "a+")
			 

# according the result calculate by function 'correlation_between_properties'
#	we delete the properties whose correlation is bigger than 0.85
def according_properties_correlation_delete():
	contents = load_result("pearsonr_spearmanr_Strong_correlation.csv")
	array_contents = np.array(contents)
	comp_fea1 = np.array(array_contents[1:, 0])
	comp_fea2 = np.array(array_contents[1:, 1])



	delete_features = [comp_fea2[i] for i in range(len(comp_fea2)) \
						if comp_fea1[i] not in comp_fea2]
	#print(set(delete_features))
	return np.array(list(set(delete_features)))

'''
scipy.stats.variation

'''
from scipy import stats
from collections import OrderedDict
def according_coefficient_variation_delete(data, features):
	waiting_to_delete = np.array(load_result("complex_value_features.csv"))
	waiting_to_delete = waiting_to_delete.reshape((waiting_to_delete.size,))
	#print(waiting_to_delete)
	indexs = get_known_features_index(features, waiting_to_delete)
	coefficient_variation_info = OrderedDict()
	for fea_pos in indexs:
		try:
			coefficient_variation_fea = stats.variation(data[:, fea_pos])
			coefficient_variation_info[features[fea_pos]] = coefficient_variation_fea
		except:
			pass
	return coefficient_variation_info


def use_RandomForestRegressor_to_delete(data, features, label):
	from sklearn.cross_validation import cross_val_score, ShuffleSplit
	from sklearn.ensemble import RandomForestRegressor

	rf = RandomForestRegressor(n_estimators=50, max_depth=4)
	scores = []
	deleted_features = list()
	for i in range(1, data.shape[1]):
	     score = cross_val_score(rf, data[:, i:i+1], label, scoring="r2",
	                              cv=ShuffleSplit(len(data), 3, .3))
	     scores.append((round(np.mean(score), 3), features[i]))
	     if round(np.mean(score), 3) < 0.01:
	     	deleted_features.append({features[i]:round(np.mean(score), 3)})
	save_result(deleted_features, "RandomForestRegressor_delete_result.csv")
	print(sorted(scores, reverse=True))


# i think that weu just delete a features as same name 
#	for example: 
#		input: needed_delete_featuers --> maybe just all the UserInfo_.. named features
def use_PCA_to_delete(data, features, needed_delete_featuers):
	stored_features = dict()
	for fea in needed_delete_featuers:

		stored = list()

		print("now!:", fea)
		fea_index = find_featuers_index(fea, features)
		print("finded: ", fea_index)
		delete_features_data = data[:, fea_index]
		from sklearn import decomposition
		pca = decomposition.PCA()
		pca.fit(delete_features_data)

		result = pca.explained_variance_
		print(result)
		mean = np.mean(result)
		print("mean:", mean)
		stored = [features[fea_index[i]] for i in range(len(result)) \
							 if result[i] >= mean]
		#print(stored)

		save_result(stored, "after_deleted_by_pca.csv", style = "a+")
		stored_features[fea] = stored
	print(stored_features)
	return stored_features 

# from minepy import MINE
# def MIC_between_features(data, features):
# 	fixed_str_features = np.array(load_result("str_features.csv"))[0]
# 	indexs = get_known_features_index(features, fixed_str_features)

# 	title = list()
# 	title.append("features1")
# 	title.append("features2")
# 	title.append("calculate_method") 
# 	title.append("MIC")
# 	save_result(title, "MIC_results.csv")
# 	save_result(title, "MIC_Strong_correlation.csv")
# 	for fea_pos in range(1, len(features)):
# 		for fea_pos_add in range(fea_pos + 1, len(features)):
# 			info_result = list()
# 			info_result.append(features[fea_pos])
# 			info_result.append(features[fea_pos_add])
# 			a1 = data[:, fea_pos]
# 			a2 = data[:, fea_pos_add]
# 			info_result.append("MIC")
# 			mic = MINE().compute_score(a1, a2)

# 			mic = round(mic, 3)
# 			info_result.append(mic)
# 			if mic >= 0.8:
# 				save_result(info_result, "MIC_results.csv", style = "a+")
# 			if mic >= 0.9:
# 				save_result(info_result, "MIC_Strong_correlation.csv", \
# 												style = "a+")

## here we use some of the ML module to compare the importance of element to the result
def extract_data_by_features(data, features, needed_features):
	needed_features_index = get_known_features_index(features, needed_features)
	new_data = np.ones((data.shape[0], len(needed_features)), dtype=np.int64)
	print(len(needed_features_index))
	print(new_data.shape)
	for i in range(len(needed_features)):
		new_data[:, i] = data[:, needed_features_index[i]]

	new_features = needed_features
	return new_data, new_features





import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
def use_forests_features_detect(data, features, label):

	# Build a forest and compute the feature importances
	forest = ExtraTreesClassifier(n_estimators=200,
	                              random_state=0)

	forest.fit(data, label)
	importances = forest.feature_importances_
	#print(importances)


	indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	print("Feature ranking:")

	for f in range(data.shape[1]):
	    print("%s. %d (%f)" % (features[indices[f]], indices[f], importances[indices[f]]))




# if a value in one features is bigger than 20000, besides
#	the positive in it is almost equal to the positive in the train data

if __name__ == '__main__':
	#################### used to calculate the correlation between properties #########
	contents = load_result("data_after_delete_no_discrimination_features.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	from map_features_to_digit import convert_to_numerical
	from solve_data import delete_features

	data = convert_to_numerical(data, features)

	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])


	correlation_between_properties(data, features)

	delete_result = according_properties_correlation_delete()
	save_result(delete_result, "deleted_features_with_strong_correlation.csv")

	
	data, features, deleted_features = delete_features(data, features, \
	 													delete_feas_list = delete_result)
	# print(deleted_features)
	save_result(data, "data_after_delete_strong_correlation_features.csv", features)
	print(data.shape)

	###############3 used pca to delete #####################

	# features_style = ["UserInfo", "WeblogInfo", "ThirdParty_Info_Period1", \
	# 				"ThirdParty_Info_Period2", "ThirdParty_Info_Period3", \
	# 				"ThirdParty_Info_Period4", "ThirdParty_Info_Period5", \
	# 				"ThirdParty_Info_Period6"]
	# #print(features)
	# # use_PCA_to_delete(data, features, features_style)


	#################### use forest to select features ###############
	# contents = load_result("after_add_new_features_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# data = convert_to_numerical(data, features)

	# label_lines = np.array(load_result("train_label_original.csv"))
	# print(label_lines.shape)
	# from save_load_result import convert_to_float
	# label = convert_to_float(label_lines)
	# use_forests_features_detect(data[:, -8:], features[-8:], label)


	# ############### use coefficient_variation to delete features ###############
	# coefficient_variation_info = according_coefficient_variation_delete(data, features)
	# print(coefficient_variation_info)

	################# use the LR module to reduce the module ####################
	# needed_features = np.array(load_result("complex_value_features.csv"))
	# needed_features = needed_features.reshape(needed_features.size, )
	# print(needed_features)
	# new_data, new_features = extract_data_by_features(data, features, needed_features)
	# print(new_data.shape)
	# print(new_features)
	# print(new_data[0])
	# from sklearn.decomposition import pca

	# Create a signal with only 2 useful dimensions
	# x1 = np.random.normal(size=100)
	# x2 = np.random.normal(size=100)
	# x3 = x1 + x2
	# X = np.c_[x3, x1, x2]
	# print(X)
	# from sklearn import decomposition
	# pca = decomposition.PCA()
	# pca.fit(X)

	# print(pca.explained_variance_)

	# print(pca.components_)

	# # As we can see, only the 2 first components are useful
	# pca.n_components = 2
	# X_reduced = pca.fit_transform(X)

	# print(X_reduced.shape)
