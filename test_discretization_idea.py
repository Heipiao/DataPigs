#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-01 16:18:22
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from save_load_result import load_result, convert_to_float
from solve_data import feature_value_class, FeatureInData

from scipy import stats


# contents = load_result("after_solve_specialStr_digited_data.csv")
# features = np.array(contents[0])
# data = np.array(contents[1:])
# data = convert_to_digit(data)

# label_lines = np.array(load_result("train_label_original.csv"))

# from save_load_result import convert_to_float, save_features_info
# label = convert_to_float(label_lines)

# #print(features)

# age_feature = "UserInfo_17"
# fea_pos = np.where(features == age_feature)[0][0]

# age_feature_info = feature_value_class(data, fea_pos, label)

# t1 = age_feature_info["average_positive"] * age_feature_info["num_positive"]
# t2 = age_feature_info["average_negitive"] * age_feature_info["num_negitive"]
# average = (t1 + t2) / (age_feature_info["num_positive"] + age_feature_info["num_negitive"])

# print(average)
# my_sum = 0
# for i in range(len(data)):
# 	my_sum += data[i, fea_pos]
# print("average = ", my_sum / len(data))

# print(np.where(data[:, fea_pos] == 0))
# print(data[2691, fea_pos])
# print(max(data[:, fea_pos]))
# print(min(data[:, fea_pos]))



data = np.array([[1,2,3,4], [5,6,7,8], [1,5,6,9], [0, 2,4,6], [4,4,4,9], [3,4,6,2], [2,1,1,2], \
				[3,6,2,9], [1,0,2,5]])
print(data)
print(data[:, 3].shape)

b = data[:, 3]
values_in_fea = np.array(list(set(list(b))))
sorted_fea_data = list(values_in_fea[np.argsort(values_in_fea)])
print(sorted_fea_data)
print(list(sorted_fea_data))

def calculate_mine(each_bin_values):
	return sum([value / 10 for value in each_bin_values])

sta_result = stats.binned_statistic(sorted_fea_data, sorted_fea_data, calculate_mine, bins = sorted_fea_data)
print(sta_result)
cal_result = sta_result[0]
bin_result = sta_result[1]
print(cal_result)
print(bin_result)


print("**** result ****")

def compare_and_combine(sta_result):
	new_edges = list()
	
	bin_edges = list(sta_result[1])
	cal_result = list(sta_result[0])
	# add the bottom of the edges
	new_edges.append(bin_edges[0])

	j = 0
	for i in range(1, len(cal_result)):
		precise_content = 0.3
		# can not combine neighbor bins
		if cal_result[j] * cal_result[i] < 0 or \
			abs(cal_result[j] - cal_result[i]) > precise_content:
			new_edges.append(bin_edges[i])
			j = i
			continue
	# add the upper of this edges
	new_edges.append(bin_edges[-1])
	return new_edges

new_bins = compare_and_combine(sta_result)
print(new_bins)

l1 = [1,2,3,4,5]
l2 = [2,3,4,5,6]
l3 = [1,2,3,4,5]
if l1 == l3:
	print("equal")

i = 0
while i < 3:
	print("ok")
	m = list()
	m.append(i)
	i += 1
print(m)