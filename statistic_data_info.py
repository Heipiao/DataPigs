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

if __name__ == '__main__':
	# contents = load_result("data_after_delete_too_many_missing_features.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# #data = convert_to_numerical(data, features)

	# print(data.shape)
	# print(features.shape)

	# label_lines = np.array(load_result("train_label_original.csv"))
	# #print(label_lines.shape)
	# from save_load_result import convert_to_float
	# label = convert_to_float(label_lines)

	# label = label.reshape((label.size, ))

	# key_feature = ["Education_Info8"]
	# t = one_features_info(data, features, label, key_feature)
	# for k, v in t.items():

	# 	print(k, v)

	
	# key_features = ["Education_Info6", "Education_Info7", "Education_Info8"]
	# print(key_features)
	# combine_info = compare_features_info(data, features, label, key_features)
	# #print(combine_info)
	# for k, v in combine_info.items():
	# 	print(k, v)




	# key_features = ["Education_Info2", "Education_Info3", "Education_Info4"]
	# print(key_features)
	# combine_info = compare_features_info(data, features, label, key_features)
	# print(combine_info)

	# map_to_zero = list()
	# map_to_one = list()
	# map_to_two = list()
	# map_to_three = list()
	# map_to_four = list()
	# for k, v in combine_info.items():
	# 	if isinstance(v, dict):
	# 		sort_v = sorted(v.items(), key=lambda d: d[1][1])
	# 		for t in sort_v:
	# 			d = t[0].split("_")
	# 			print(d, t[1])
	# 			if t[1][1] == 0:
	# 				map_to_zero.append(d)
	# 			elif t[1][1] == 1:
	# 				map_to_one.append(d)
	# 			elif t[1][1] < 10:
	# 				map_to_two.append(d)
	# 			elif t[1][1] < 20:
	# 				map_to_three.append(d)
	# 			if d[0] == "E":
	# 				map_to_four.append(d)
	# print(map_to_zero)
	# print(map_to_one)
	# print(map_to_two)
	# print(map_to_three)
	# print(map_to_four)




	# key_features = ["UserInfo_22", "UserInfo_23"]
	# print(key_features)
	# combine_info = compare_features_info(data, features, label, key_features)
	# print(combine_info)
	
	# map_to_zero = list()
	# map_to_one = list()
	# map_to_two = list()
	# map_to_three = list()
	# map_to_four = list()
	# map_to_five = list()
	# for k, v in combine_info.items():
	# 	print(k, v)
	# 	if isinstance(v, dict):
	# 		sort_v = sorted(v.items(), key=lambda d: d[1][1])
	# 		user23_iii = list()
	# 		for t in sort_v:
	# 			d = t[0].split("_")
	# 			print(d, t[1])
				# if d[0] == "-1" and d[1] == "-1":
				# 	map_to_five.append(d)
				# 	continue
				# if t[1][1] == 0:
				# 	map_to_zero.append(d)
				# elif t[1][1] < 10:
				# 	map_to_one.append(d)
				# elif t[1][1] < 20:
				# 	map_to_two.append(d)
				# elif t[1][1] < 100:
				# 	map_to_three.append(d)
				# else:
				# 	map_to_four.append(d)
				# print(t[0], t[1])
	# 			if d[0] == "-1" and d[1] == "-1":
	# 				map_to_five.append(d)
	# 				continue
	# 			if t[1][0] < 10:
	# 				map_to_zero.append(d)
	# 			elif t[1][0] < 20:
	# 				map_to_one.append(d)
	# 			elif t[1][0] < 100:
	# 				map_to_two.append(d)
	# 			elif t[1][0] < 1000:
	# 				map_to_three.append(d)
	# 			else:
	# 				map_to_four.append(d)
	# 			#print(t[0], t[1])
	# print(map_to_zero)
	# print(map_to_one)
	# print(map_to_two)
	# print(map_to_three)
	# print(map_to_four)
	# print(map_to_five)

	# key_feature = ["WeblogInfo_20"]
	# t = one_features_info(data, features, label, key_feature)
	# map_to_zero = list()
	# map_to_one = list()
	# map_to_two = list()
	# map_to_three = list()
	# map_to_four = list()
	# map_to_five = list()
	# sort_v = sorted(t.items(), key=lambda d: d[1][1])
	# for v in sort_v:
	# 	print(v[0], v[1])
	# 	# if v[0] == "-1":
	# 	# 	map_to_five.append(v[0])
	# 	# 	continue
	# 	# if v[1][0] < 20:
	# 	# 	map_to_zero.append(v[0])
	# 	# elif v[1][0] < 100:
	# 	# 	map_to_one.append(v[0])
	# 	# elif v[1][0] < 1000:
	# 	# 	map_to_two.append(v[0])
	# 	# elif v[1][0] < 5000:
	# 	# 	map_to_three.append(v[0])
	# 	# else:
	# 	# 	map_to_four.append(v[0])




	# 	if v[0] == "-1":
	# 		map_to_five.append(v[0])
	# 		continue
	# 	if v[1][1] == 0:
	# 		map_to_zero.append(v[0])
	# 	elif v[1][1] < 10:
	# 		map_to_one.append(v[0])
	# 	elif v[1][1] < 100:
	# 		map_to_two.append(v[0])
	# 	elif v[1][1] < 200:
	# 		map_to_three.append(v[0])
	# 	else:
	# 		map_to_four.append(v[0])
	# print(map_to_zero)
	# print(map_to_one)
	# print(map_to_two)
	# print(map_to_three)
	# print(map_to_four)
	# print(map_to_five)
	contents = load_result("data_after_thirdparty_solved.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])

	data = convert_to_numerical(data, features)
	label_lines = np.array(load_result("train_label_original.csv"))
	#print(label_lines.shape)
	from save_load_result import convert_to_float
	label = convert_to_float(label_lines)

	label = label.reshape((label.size, ))

	from view_data import view_tp_cv
	# type_sta_info = sta_thirdParty_info(data, features, ["9"], label)
	# # m = 0
	# # for k, v in type_sta_info.items():
	# # 	m += 1
	# # 	if m < 500:
	# # 		for sub_k, sub_v in v.items():
	# # 			print(sub_k, sub_v)
	# sat_type = "ms_type9"
	# sat_name = "average"
	# res = [v[sat_type][sat_name] for v in type_sta_info.values()]
	# ave_res = [value for value in res if not value == 0]
	# print("max: ", max(res))
	# print("min: ", min(res))
	# view_tp_cv(type_sta_info, sat_type, sat_name, saved_fig = sat_type + sat_name + ".png")



	# from create_features_from_thirdpart import bin_each_type
	# type_numbers = [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"],
	# 			["10"], ["11"], ["12"], ["13"], ["14"], ["15"], ["16"], ["17"]]
	# for type_number in type_numbers:
	# 	type_sta_info = sta_thirdParty_info(data, features, type_number, label)

	# 	# m = 0
	# 	# for k, v in type_sta_info.items():
	# 	# 	m += 1
	# 	# 	if m < 500:
	# 	# 		for sub_k, sub_v in v.items():
	# 	# 			print(sub_k, sub_v)

	# 	# period_number = ["1"]
	# 	# print(thirdParty_one_period_info(data, features, label, period_number))
	# 	sat_type = "ms_type" + type_number[0]
	# 	sat_name = "average"
	# 	print(sat_type)
	# 	res = [v[sat_type][sat_name] for v in type_sta_info.values()]
	# 	ave_res = [value for value in res if not value == 0]
	# 	binned_res = bin_each_type(ave_res)
	# 	view_tp_cv(type_sta_info, sat_type, sat_name, saved_fig = sat_type + sat_name + ".png")
		
	# print("result")
	# print(binned_res[0])
	# print(binned_res[1])
	# print(binned_res[2][:100])

	from create_features_from_thirdpart import third_party_level
	data, features = third_party_level(data, features)