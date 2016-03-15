#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-08 11:34:09
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import numpy as np

from solve_data import feature_value_class, FeatureInData
from scipy import stats
from save_load_result import load_result, convert_to_float



def discretization_feature(fea_info, needed_bin_values = None):
	# user do not give values to make bins
	# load the values in fea_info
	if needed_bin_values == None:
		needed_bin_values = [k for k in fea_info.keys() \
								if isinstance(fea_info[k], FeatureInData)]
	# move the repeated values, and sort eh 
	values_in_fea = np.array(list(set(list(needed_bin_values))))
	sorted_to_bin_values = values_in_fea[np.argsort(values_in_fea)]

	# after the sentence above, the sorted_to_bin_values
	#	will be a array whose size is (a, )
	# making a list to contain the bins
	# and our initialed bins are all the values

	def weight_positive(fea_value):
		return (fea_info[fea_value]._respond_positive_num / fea_info[fea_value]._present_num) * \
				(fea_info[fea_value]._respond_positive_num / fea_info["num_positive"])

	def weight_negitive(fea_value):
		return (fea_info[fea_value]._respond_negitive_num / fea_info[fea_value]._present_num) * \
				(fea_info[fea_value]._respond_negitive_num / fea_info["num_negitive"])
	def bin_label(bin_values):
		y1 = round(np.mean([round(weight_positive(bin_value), 4) for bin_value in bin_values]), 4)
		y2 = round(np.mean([round(weight_negitive(bin_value), 4) for bin_value in bin_values]), 4)
		# choose the most notable as its label

		if y1 > y2:
			return y1
		else:
			return -1 * y2
	# sta_result[0] is calculate result for each bin
	# sta_result[1] is bin_edges
	# sta_result[2] is label of each value under bins
	# this function will return a combine bins
	def compare_and_combine(sta_result):
		new_edges = list()
		
		bin_edges = list(sta_result[1])
		cal_result = list(sta_result[0])
		print(bin_edges)
		print(cal_result)
		# add the bottom of the edges
		new_edges.append(bin_edges[0])

		j = 0
		for i in range(1, len(cal_result)):
			precise_content = min(cal_result[j], cal_result[i])# / 10
			# can not combine neighbor bins
			if cal_result[j] * cal_result[i] < 0 or \
				abs(cal_result[j] - cal_result[i]) > abs(precise_content):
				new_edges.append(bin_edges[i])
				j = i
				continue
		# add the upper of this edges
		new_edges.append(bin_edges[-1])
		return new_edges



	bins = list(sorted_to_bin_values)
	print(bins)
	iteral = 0

	while iteral < 100:
		sta_result = stats.binned_statistic(sorted_to_bin_values, sorted_to_bin_values, \
										bin_label, bins)

		new_bins = compare_and_combine(sta_result)
		if bins == new_bins:
			break
		bins = new_bins

	return sta_result

if __name__ == '__main__':
	contents = load_result("after_Str_features_digited_data.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])
	label_lines = np.array(load_result("train_label_original.csv"))
	print(label_lines.shape)

	label = convert_to_float(label_lines)

	from map_features_to_digit import convert_to_numerical

	data = convert_to_numerical(data, features)

	#index = np.where(features == "ThirdParty_Info_Period4_1")[0][0]
	index = np.where(features == "WeblogInfo_12")[0][0]
	fea_info = feature_value_class(data, index, label)
	#print(fea_info)
	result = discretization_feature(fea_info)
	print(result)