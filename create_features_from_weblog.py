#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-18 14:09:11
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
from save_load_result import save_result, load_result
from map_features_to_digit import convert_to_numerical
from solve_data import get_known_features_index
from solve_data import delete_features
import numpy as np
from functools import reduce
from collections import OrderedDict

from create_new_features import find_featuers_index

from features_reduce import correlation_between_properties, according_properties_correlation_delete


def new_WI_19(data, features):
	solved_features = ["WeblogInfo_19"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "WeblogInfo_19_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "H":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "G":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "J":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "E":
			feature_data[user, 0] = 3
		elif data[user, fea_indexs[0]] == "F":
			feature_data[user, 0] = 4
		elif data[user, fea_indexs[0]] == "D":
			feature_data[user, 0] = 5
		else:
			feature_data[user, 0] = 6

	
	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
								delete_feas_list = solved_features)

	print("WeblogInfo_19 solved")
	print(deleted)
	return new_data, new_features


def new_WI_20_by_present(data, features):
	solved_features = ["WeblogInfo_20"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "WeblogInfo_20_present_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))
	none_finded_combine = OrderedDict()
	feature_data = np.zeros((len(data), 1))
	map_to_zero = ['F3', 'C39', 'F5', 'F2', 'F9', 'F12', 'I8', 'C38', 'F10', 'F11', 'O', 'C13', 'I6', 'C16', 'I7', 'I10']
	map_to_one = ['F14', 'F15', 'F13', 'C17', 'C12', 'C18', 'C15']
	map_to_two = ['F16', 'C11', 'C1', 'C20', 'C19']
	map_to_three = ['I3', 'U', 'C21', 'I4']
	map_to_four = ['I5']
	map_to_five = ['-1']
	for user in range(data.shape[0]):
		fea_value = data[user, fea_indexs[0]]
		if fea_value in map_to_zero:
			feature_data[user, 0] = 0
		elif fea_value in map_to_one:
			feature_data[user, 0] = 1
		elif fea_value in map_to_two:
			feature_data[user, 0] = 2
		elif fea_value in map_to_three:
			feature_data[user, 0] = 3
		elif fea_value in map_to_four:
			feature_data[user, 0] = 4
		elif fea_value in map_to_five:
			feature_data[user, 0] = 5
		else:
			# print("error")
			# print(fea_value)
			if fea_value not in none_finded_combine.keys():
				none_finded_combine[fea_value] = list()
			none_finded_combine[fea_value].append(user)

	for fea_value, users in none_finded_combine.items():
		if fea_value[0] == "-1":
			feature_data[users, 0] = 5
		if len(users) < 20:
			feature_data[users, 0] = 0
		elif len(users) < 100:
			feature_data[users, 0] = 1
		elif len(users) < 1000:
			feature_data[users, 0] = 2
		elif len(users) < 5000:
			feature_data[users, 0] = 3
		else:
			feature_data[users, 0] = 4

	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
 								delete_feas_list = solved_features)
	print("WeblogInfo_20 solved present")
	print(deleted)
	return new_data, new_features

# def new_WI_20_by_positive(data, features):
# 	solved_features = ["WeblogInfo_20"]
# 	fea_indexs = get_known_features_index(features, solved_features)

# 	feature_name = "WeblogInfo_20_positive_info_(cat)"
# 	new_add_feature = np.array([feature_name])
# 	new_features = np.concatenate((features, new_add_feature))
# 	none_finded_combine = OrderedDict()
# 	feature_data = np.zeros((len(data), 1))
# 	map_to_zero = ['I8', 'O', 'F15', 'I6', 'F12', 'C16', 'F5', 'F11', 'F3', 'F10', 'C39']
# 	map_to_one = ['C13', 'C38', 'F2', 'C17', 'F9', 'I7', 'F13', 'I10', 'C12', 'F14', 'C18', 'C15']
# 	map_to_two = ['F16', 'C11', 'C1', 'C20', 'C19', 'I3', 'U']
# 	map_to_three = ['C21', 'I4']
# 	map_to_four = ['I5']
# 	map_to_five = ['-1']
# 	for user in range(data.shape[0]):
# 		if data[user, fea_indexs[0]] in map_to_zero:
# 			feature_data[user, 0] = 0
# 		elif data[user, fea_indexs[0]] in map_to_one:
# 			feature_data[user, 0] = 1
# 		elif data[user, fea_indexs[0]] in map_to_two:
# 			feature_data[user, 0] = 2
# 		elif data[user, fea_indexs[0]] in map_to_three:
# 			feature_data[user, 0] = 3
# 		elif data[user, fea_indexs[0]] in map_to_four:
# 			feature_data[user, 0] = 4
# 		elif data[user, fea_indexs[0]] in map_to_five:
# 			feature_data[user, 0] = 5
# 		else:
# 			print("error")
# 			if fea_value not in none_finded_combine.keys():
# 				none_finded_combine[fea_value] = list()
# 			none_finded_combine[fea_value].append(user)

# 	for fea_value, users in none_finded_combine.items():
# 		if fea_value[0] == "-1":
# 			feature_data[users, 0] = 5
# 		if len(users) < 10:
# 			feature_data[users, 0] = 0
# 		elif len(users) < 100:
# 			feature_data[users, 0] = 1
# 		elif len(users) < 1000:
# 			feature_data[users, 0] = 2
# 		elif len(users) < 5000:
# 			feature_data[users, 0] = 3
# 		else:
# 			feature_data[users, 0] = 4

# 	new_data = np.concatenate((data, feature_data), axis = 1)
# 	new_data, new_features, deleted = delete_features(new_data, new_features, \
# 								delete_feas_list = solved_features)

# 	print("WeblogInfo_20 solved by positive")
# 	print(deleted)
# 	return new_data, new_features

def new_WI_21(data, features):
	solved_features = ["WeblogInfo_21"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "WeblogInfo_21_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "B":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "C":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "A":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "D":
			feature_data[user, 0] = 3
		else:
			feature_data[user, 0] = 4

	
	new_data = np.concatenate((data, feature_data), axis = 1)

	new_data, new_features, deleted = delete_features(new_data, new_features, \
									delete_feas_list = solved_features)


	print("WeblogInfo_21 solved")
	print(deleted)
	return new_data, new_features


def solve_weblog_info_package(data, features, saved_dir = "resultData/"):
	from map_features_to_digit import convert_to_numerical
	from solve_data import delete_features

	data, features = new_WI_19(data, features)
	data, features = new_WI_20_by_present(data, features)
	#data, features = new_WI_20_by_positive(data, features)
	data, features = new_WI_21(data, features)
#	save_result(data, "data_after_solve_WeblogInfo_21.csv", features, dir_name = saved_dir)
	
	save_result(data, "data_after_solved_weblog.csv", features, dir_name = saved_dir)

	return data, features

if __name__ == '__main__':
	contents = load_result("data_after_solved_UserInfo22_23.csv", dir_name = saved_dir)
	features = np.array(contents[0])
	data = np.array(contents[1:])

	#data = convert_to_numerical(data, features)

	print(data.shape)
	print(features.shape)

	from create_new_features import find_featuers_index
	features_name = "WeblogInfo"
	fea_indexs = find_featuers_index(features_name, features)
	print(fea_indexs)
	weblog_data = data[:, fea_indexs]
	weblog_features = features[fea_indexs]

	print(weblog_data.shape)
	print(weblog_features.shape)
	#save_result(weblog_data, "weblog_data_view.csv", weblog_features)

	# # label_lines = np.array(load_result("train_label_original.csv"))
	# # #print(label_lines.shape)
	# # from save_load_result import convert_to_float
	# # label = convert_to_float(label_lines)

	# # label = label.reshape((label.size, ))
	correlation_between_properties(weblog_data, weblog_features)
	delete_result = according_properties_correlation_delete()
	save_result(delete_result, "deleted_weblog_features_with_strong_correlation.csv")
	weblog_data, weblog_features, deleted_features = delete_features(weblog_data, weblog_features, \
 													delete_feas_list = delete_result)
	save_result(weblog_data, "data_after_delete_strong_correlation_weblog.csv", weblog_features)


	weblog_delete_needed = ["WeblogInfo_10", "WeblogInfo_11", "WeblogInfo_12", "WeblogInfo_13",
							"WeblogInfo_23", "WeblogInfo_25", "WeblogInfo_26", "WeblogInfo_28",
							"WeblogInfo_31", "WeblogInfo_55", "WeblogInfo_58"]
	save_result(weblog_delete_needed, "deleted_useless_weblog.csv")

	new_data, new_features, deleted = delete_features(weblog_data, weblog_features, \
								delete_feas_list = weblog_delete_needed)
	save_result(new_data, "data_after_delete_useless_weblog.csv", new_features)