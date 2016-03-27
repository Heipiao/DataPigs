#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-19 15:24:31
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

from scipy import stats

from sklearn.preprocessing import Imputer

def fill_thirdParty_miss(data, features):

	based_feature_name = "ThirdParty_Info_Period"
	third_p_index = find_featuers_index(based_feature_name, features)
	
	imp = Imputer(missing_values=-1, strategy='median', axis=0)
	# tp_data = data[:, third_p_index]
	# tp_features = features[third_p_index]

	data[:, third_p_index] = imp.fit(data[:, third_p_index]).transform(data[:, third_p_index])
	return data, features
	#save_result(tp_data, "tp_data_after_filling_miss.csv", tp_features)

# no miss: 0
# miss in first period: 1
# ... so on
def get_first_missing_period(user_data, features):
	period_one = "ThirdParty_Info_Period1"
	period_one_index = find_featuers_index(period_one, features)
	period_two = "ThirdParty_Info_Period2"
	period_two_index = find_featuers_index(period_two, features)
	period_three = "ThirdParty_Info_Period3"
	period_three_index = find_featuers_index(period_three, features)
	period_four = "ThirdParty_Info_Period4"
	period_four_index = find_featuers_index(period_four, features)
	period_five = "ThirdParty_Info_Period5"
	period_five_index = find_featuers_index(period_five, features)
	period_six = "ThirdParty_Info_Period6"
	period_six_index = find_featuers_index(period_six, features)
	if -1 in list(user_data[period_one_index]):
		return 1
	elif -1 in list(user_data[period_two_index]):
		return 2
	elif -1 in list(user_data[period_three_index]):
		return 3
	elif -1 in list(user_data[period_four_index]):
		return 4
	elif -1 in list(user_data[period_five_index]):
		return 5
	elif -1 in list(user_data[period_six_index]):
		return 6
	else:
		return 0
	return 0
def sta_start_missing_period(data, features):

	feature_name = "ThirdParty_missing_period_count"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		feature_data[user, 0] = get_first_missing_period(data[user, :], features)


	new_data = np.concatenate((data, feature_data), axis = 1)

	print("missing peroid counted")
	#save_result(new_data, "data_after_counted_tp.csv", new_features)
	return new_data, new_features

def remove_thirdparty6(data, features):
	solved_features = "ThirdParty_Info_Period6"
	third_p6_index = find_featuers_index(solved_features, features)


	new_data, new_features, deleted = delete_features(data, features, \
								delete_fea_pos = third_p6_index)

	print("ThirdParty_Info_Period6 all removed")
	print(deleted)
	return new_data, new_features

# ThirdParty_Info_Period2_3
# input:
#	- calculate_number: the type you want to view --> just same as: calculate_number = ["17"]
def third_party_stable(data, features):

	type_numbers = [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"],
					["10"], ["11"], ["12"], ["13"], ["14"], ["15"], ["16"], ["17"]]
	for type_num in type_numbers:
		from statistic_data_info import sta_thirdParty_info
		type_sta_info = sta_thirdParty_info(data, features, type_num)

		feature_name = "ThirdParty" + "*" + "_" + type_num[0] + " stable"
		sta_name = "ms_type" + type_num[0]
		print(feature_name + "....")
		new_add_feature = np.array([feature_name])
		features = np.concatenate((features, new_add_feature))

		feature_data = np.zeros((len(data), 1))

		for user in range(data.shape[0]):
			user_stable = type_sta_info[user][sta_name]["cv"]
			if user_stable == 0:
				feature_data[user, 0] = 0
			elif user_stable < 0.5:
				feature_data[user, 0] = 1
			elif user_stable < 1:
				feature_data[user, 0] = 2
			elif user_stable < 1.5:
				feature_data[user, 0] = 3
			elif user_stable < 2:
				feature_data[user, 0] = 4
			elif user_stable == 2:
				feature_data[user, 0] = 5
			else:
				print("error!!!!....")
		data = np.concatenate((data, feature_data), axis = 1)

	return data, features

def compare_and_combine(sta_result):
	new_edges = list()
	
	cal_result = list(sta_result[0])
	bin_edges = list(sta_result[1])

	# print(bin_edges)
	# print(cal_result)
	# add the bottom of the edges
	new_edges.append(bin_edges[0])

	j = 0
	for i in range(1, len(cal_result)):
		precise_content = 30
		# can not combine neighbor bins
		if abs(cal_result[j] - cal_result[i]) > precise_content:
			new_edges.append(bin_edges[i])
			j = i
			continue
	# add the upper of this edges
	new_edges.append(bin_edges[-1])
	new_edges = [round(new_edges[i], 2) for i in range(len(new_edges))]
	return new_edges

# after satistic each user`s each type(1~17) averagea
# 	we bin thetype
def bin_each_type(data_to_bin):
	#stats.binned_statistic()
	bin_num = int(np.max(data_to_bin) / np.mean(data_to_bin))
	counted, bin_edges, bin_label = stats.binned_statistic(data_to_bin, data_to_bin, 'count', bins=bin_num)

	while 1:
		binned_result = stats.binned_statistic(data_to_bin, data_to_bin, 'count', bins=bin_edges)
		new_bins_edge = compare_and_combine(binned_result)

		if len(new_bins_edge) == len(bin_edges):
			break
		else:
			bin_edges = new_bins_edge


	return binned_result

		
def third_party_level(data, features):
	type_numbers = [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"],
				["10"], ["11"], ["12"], ["13"], ["14"], ["15"], ["16"], ["17"]]
	for type_num in type_numbers:
		from statistic_data_info import sta_thirdParty_info
		type_sta_info = sta_thirdParty_info(data, features, type_num)

		sat_type = "ms_type" + type_num[0]
		sat_name = "average"
		res = [v[sat_type][sat_name] for v in type_sta_info.values()]
		ave_res = [value for value in res if not value == 0]
		binned_res = bin_each_type(ave_res)
		type_bin_rule = binned_res[1]


		feature_name = "ThirdParty" + "*" + "_" + type_num[0] + " level"

		sta_name = "ms_type" + type_num[0]
		print(feature_name + "....")
		new_add_feature = np.array([feature_name])
		features = np.concatenate((features, new_add_feature))

		feature_data = np.zeros((len(data), 1))
		for user in range(data.shape[0]):
			user_average = type_sta_info[user][sat_type][sat_name]
			if user_average == 0:
				feature_data[user, 0] = 0
				continue
			feature_data[user, 0] = np.digitize(user_average, type_bin_rule)

		data = np.concatenate((data, feature_data), axis = 1)

	return data, features



def solve_thirdparty_info_package(data, features, saved_dir = "resultData/"):
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	data = convert_to_numerical(data, features)

	data, features = sta_start_missing_period(data, features)
	data, features = remove_thirdparty6(data, features)

	data, features = fill_thirdParty_miss(data, features)

	data, features = third_party_stable(data, features)

	data, features = third_party_level(data, features)
	save_result(data, "data_after_thirdparty_solved.csv", features, dir_name = saved_dir)
	return data, features 

if __name__ == '__main__':

	contents = load_result("data_after_solved_weblog.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])

	data = convert_to_numerical(data, features)

	solve_thirdparty_info_package(data, features)

	# calculate_number = ["17"]
	# users_sta_name, users_stability = calculate_stability(data, features, calculate_number[0])
	# print(users_sta_name)
	# for i in range(10):
	# 	print(users_stability[i])
	# from create_new_features import find_featuers_index
	# features_name = "ThirdPart"
	# fea_indexs = find_featuers_index(features_name, features)
	# print(fea_indexs)

	# data = data[:, fea_indexs]
	# features = features[fea_indexs]
	# save_result(data, "data_of_ThirdPart.csv", features)
	