#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-15 15:38:52
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import numpy as np
from collections import OrderedDict
from save_load_result import load_result
from map_features_to_digit import convert_to_numerical
from solve_data import get_known_features_index
from solve_data import delete_features
from functools import reduce
# this function aim to statistic how many samples in 
#	data have missed features and their position
# input: 
#	- data: all the instances --> should contain the idx as fisrt col
#	- label: respond label

# output:
#	missed_infos: the whole missed info 
#		- missed_instances_sum: the total number of missed
#		- positive_sum: positive target instances in missed
#		- negitive_sum: inverse above
#	users_miss_info: the detail of user`s missed information
#		- idx: the idx of the instances
#			- missed_count: how many the user missed
#			- miss_features_indexs: what the user missed
#			- missed_features: what the user missed
#			- label: the label of this user 

def missed_instances_info(data, features, label, key_feature = None):
	missed_infos = OrderedDict()
	users_miss_info = OrderedDict()


	if key_feature == None:
		indexs = range(len(features))
		key_data = data
		key_feature = features
	else:
		indexs = get_known_features_index(features, key_feature)
		key_data = data[:, indexs]
		print(key_data.shape)
		print(key_data.shape[0])
		print(key_data.shape[1])

	missed_infos["missed_instances_sum"] = 0
	missed_infos["positive_sum"] = 0
	missed_infos["negitive_sum"] = 0

	for i in range(key_data.shape[0]):
		user_miss_count = 0
		missed_features_index = list()
		missed_features = list()
		flag = 0
		user_idx = data[i, 0]
		users_miss_info[user_idx] = OrderedDict()

		for j in range(key_data.shape[1]):
			if key_data[i, j] == -1:
				flag = 1
				user_miss_count += 1
				missed_features_index.append(indexs[j])
				missed_features.append(key_feature[j])
		# exist miss in this line
		if flag == 1:
			missed_infos["missed_instances_sum"] += 1
			if label[i] == 1:
				missed_infos["positive_sum"] += 1
			else:
				missed_infos["negitive_sum"] += 1

		# if this uer missed, statistic information
		if user_miss_count:
			users_miss_info[user_idx]["missed_count"] = user_miss_count
			users_miss_info[user_idx]["miss_features_indexs"] = missed_features_index
			users_miss_info[user_idx]["missed_features"] = missed_features
			users_miss_info[user_idx]["label"] = label[i]

	return missed_infos, users_miss_info

def compare_features_info(data, features, label, key_features):
	fea_indexs = get_known_features_index(features, key_features)
	compare_result = OrderedDict()
	compare_result["num_differ"] = 0
	compare_result["different_combine_info"] = dict()
	compare_result["num_differ_positive"] = 0
	compare_result["num_same"] = 0
	compare_result["same_combine_info"] = dict()
	compare_result["num_same_positive"] = 0
	compare_result["num_same_miss"] = 0
	for user in range(data.shape[0]):
		# user_id = data[user, 0]
		combine_data = reduce(lambda x,y: str(x) + '_' + str(y), list(data[user, fea_indexs]))
		province = list(data[user, fea_indexs])[0]
		pro_info = list(data[user, fea_indexs])[1]
		if not len(set(list(data[user, fea_indexs]))) == 1:
			compare_result["num_differ"] += 1
			if not combine_data in compare_result["different_combine_info"].keys():
				compare_result["different_combine_info"][combine_data] = [0, 0]

			compare_result["different_combine_info"][combine_data][0] += 1

			if label[user] == 1:
				compare_result["num_differ_positive"] += 1
				compare_result["different_combine_info"][combine_data][1] += 1
		else:
			compare_result["num_same"] += 1

			if not combine_data in compare_result["same_combine_info"].keys():
				compare_result["same_combine_info"][combine_data] = [0, 0]

			compare_result["same_combine_info"][combine_data][0] += 1

			if label[user] == 1:
				compare_result["num_same_positive"] += 1
				compare_result["same_combine_info"][combine_data][1] += 1

			if str(-1) == list(data[user, fea_indexs])[0]:
				compare_result["num_same_miss"] += 1
	return compare_result


def one_features_info(data, features, label, key_feature):
	fea_index = get_known_features_index(features, key_feature)

	user_value_info = dict()
	for user in range(data.shape[0]):
		value = list(data[user, fea_index])[0]
		if not value in user_value_info.keys():
			user_value_info[value] = [0, 0]
		user_value_info[value][0] += 1
		if label[user] == 1:
			user_value_info[value][1] += 1
	return user_value_info


def sta_thirdParty_info(data, features, type_number, label = None):

	based_feature_name = "ThirdParty_Info_Period"
	solved_features = list()
	for i in range(1, 7):
		solved_feature = based_feature_name + str(i) + "_" + type_number[0]

		solved_features.append(solved_feature)

	indexs = get_known_features_index(features, solved_features)

	# print(solved_features)
	# print(indexs)

	users_sta_name = ["average", "Standard deviation", "coefficient of variation", "max", "min"]

	sta_name = "ms_type" + type_number[0]
	#print("sta_name: ", sta_name)
	users_stability = OrderedDict()

	for user in range(data.shape[0]):
		users_stability[user] = OrderedDict()
		if not label == None:
			users_stability[user]["label"] = label[user]
		calculate_data = data[user, indexs]
		users_stability[user]["value"] = list(calculate_data)

		users_stability[user][sta_name] = dict()
		fea_average = round(float(np.mean(calculate_data)), 3)
		fea_std = round(float(np.std(calculate_data)), 3)
		if fea_average == 0:
			fea_cv = 0
		else:
			fea_cv = round(float(fea_std / fea_average), 3)
		max_v = np.amax(calculate_data)
		min_v = np.amin(calculate_data)

		users_stability[user][sta_name]["average"] = fea_average
		users_stability[user][sta_name]["Standard deviation"] = fea_std
		users_stability[user][sta_name]["cv"] = fea_cv
		users_stability[user][sta_name]["max"] = max_v
		users_stability[user][sta_name]["min"] = min_v


	return users_stability

def thirdParty_one_period_info(data, features, label, period_number):

	based_feature_name = "ThirdParty_Info_Period"
	solved_features = list()

	for i in range(1, 17):
		solved_feature = based_feature_name + period_number[0] + "_" + str(i) 

		solved_features.append(solved_feature)

	indexs = get_known_features_index(features, solved_features)
	one_period_info = OrderedDict()
	one_period_info["missing count"] = 0
	one_period_info["missing contain positive count"] = 0
	one_period_info["missing indexs"] = list()
	one_period_info["missing indexs label"] = list()

	for user in range(data.shape[0]):
		sat_data = list(data[user, indexs])
		if -1 in sat_data:
			one_period_info["missing count"] += 1
			if label[user] == 1:
				one_period_info["missing contain positive count"] += 1
			one_period_info["missing indexs"].append(user)
			one_period_info["missing indexs label"].append(label[user])

	return one_period_info

def one_features_info2(data, features, key_feature):
	fea_index = get_known_features_index(features, key_feature)

	user_value_info = dict()
	for user in range(data.shape[0]):
		value = list(data[user, fea_index])[0]
		if not value in user_value_info.keys():
			user_value_info[value] = 0
		user_value_info[value] += 1

	return user_value_info

def compare_features_info2(data, features, key_features):
	fea_indexs = get_known_features_index(features, key_features)
	compare_result = OrderedDict()

	for user in range(data.shape[0]):
		# user_id = data[user, 0]
		combine_data = reduce(lambda x,y: str(x) + '_' + str(y), list(data[user, fea_indexs]))
		if combine_data not in compare_result.keys():
			compare_result[combine_data] = 0
		compare_result[combine_data] += 1
	return compare_result

if __name__ == '__main__':
	contents = load_result("PPD_Userupdate_Info_3_1_Training_Set.csv", dir_name = "PPD-First-Round-Data/Training Set/")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	#data = convert_to_numerical(data, features)

	print(data.shape)
	print(features.shape)

	# label_lines = np.array(load_result("train_label_original.csv"))
	#print(label_lines.shape)
	# from save_load_result import convert_to_int
	# label = convert_to_int(label_lines)

	# label = label.reshape((label.size, ))

	# key_feature = ["LogInfo2"]
	# res = one_features_info2(data, features, key_feature)
	# res = sorted(res.items(), key=lambda d: d[1]) 
	# print(res)
	# for k, v in res:
	# 	print(k, v)


	# key_features = ["LogInfo1", "LogInfo2"]
	# res = compare_features_info2(data, features, key_features)
	# for k, v in res.items():
	# 	print(k, v)

	key_feature = ["UserupdateInfo1"]
	res = one_features_info2(data, features, key_feature)
	#print(res)
	key_list = list()
	for k, v in res.items():
		key_list.append(k[1:].lower())
		#print(k, v)
	#print(key_list)
	print(set(key_list))