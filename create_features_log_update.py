#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-25 21:01:12
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
from save_load_result import save_result, load_result
import pickle
import numpy as np
from create_features_from_land_operate_info import create_features_from_log_info
from create_features_from_modify_info import create_features_from_update_info
from solve_data import delete_features
from map_features_to_digit import convert_to_numerical

from collections import OrderedDict

def load_all_id_info(for_train = True):
	if for_train:
		saved_dir = "resultData"
	else:
		saved_dir = "resultData/test"

	combined_log_update_file = os.path.join(saved_dir, "all_id_info.pickle")
	fid=open(combined_log_update_file, "rb")

	combine_data=OrderedDict(pickle.load(fid))
	print(len(combine_data))
	fid.close()

	return combine_data

		# feature_name = "ThirdParty" + "*" + "_" + type_num[0] + " stable"
		# sta_name = "ms_type" + type_num[0]
		# print(feature_name + "....")
		# new_add_feature = np.array([feature_name])
		# features = np.concatenate((features, new_add_feature))

		# feature_data = np.zeros((len(data), 1))
		
		# data = np.concatenate((data, feature_data), axis = 1)
def add_features_from_log(data, features, for_train = True):
	print("***** extract features from log information")
	combine_data = load_all_id_info(for_train)

	users_land_features = create_features_from_log_info(combine_data)
	# print(len(users_land_features))

	i = 0
	log_features = list()
	for k, v in users_land_features.items():
		if i == 0:
			for f, fv in v.items():
				log_features.append(f)
			break
	# print(log_features)

	for f in log_features:
		feature_name = f
		new_add_feature = np.array([feature_name])
		features = np.concatenate((features, new_add_feature))

		feature_data = np.zeros((len(data), 1))
		n = 0
		for user in users_land_features.keys():
			feature_data[n, 0] = users_land_features[user][f]
			n += 1

		data = np.concatenate((data, feature_data), axis = 1)

	# print(data.shape)
	# print(features.shape)
	print("**** finished *****")
	return data, features

def add_features_from_update(data, features, for_train = True):
	print("***** extract features from update information")
	combine_data = load_all_id_info(for_train)

	users_update_features = create_features_from_update_info(combine_data)

	i = 0
	update_features = list()
	for k, v in users_update_features.items():
		if i == 0:
			for f, fv in v.items():
				update_features.append(f)
			break
	# print(update_features)
	for f in update_features:
		feature_name = f
		new_add_feature = np.array([feature_name])
		features = np.concatenate((features, new_add_feature))

		feature_data = np.zeros((len(data), 1))
		n = 0
		for user in users_update_features.keys():
			feature_data[n, 0] = users_update_features[user][f]
			n += 1

		data = np.concatenate((data, feature_data), axis = 1)

	print("**** finished *****")
	# print(data.shape)
	# print(features.shape)
	return data, features


def extract_log_update_package(data, features, for_train = True):
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	data = convert_to_numerical(data, features)

	data, features = add_features_from_log(data, features, for_train)
	data, features = add_features_from_update(data, features, for_train)

	return data, features

if __name__ == '__main__':
	pass
	# contents = load_result("data_after_features_processed.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
	# data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])

	# data = convert_to_numerical(data, features)

	# extract_log_update_package(data, features, for_train = False)