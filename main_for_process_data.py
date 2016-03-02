#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-01 20:30:11
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

from get_data import load_data, data_info, print_instance_format, extract_target 
from solve_data import feature_value_class, filling_miss_with_current_set, \
						filling_miss_with_experience, delete_features, \
						sort_features_with_entroy, delete_no_discrimination_features

from map_features_to_digit import digit_city_features, digit_province_features, \
								digit_phone_features, digit_marry_features, \
								digit_resident_features, map_str_to_digit

from save_load_result import save_result, save_features_info, \
							load_result, write_to_deleted_features_area
import numpy as np

# first setup, use this function to load the data
# input:
#	- file_name: the data file`s name you want to load...
#	- for_train: default True --> get file from 'Training Set'
#							  --> get data, features, label and store all 
#				if set False  --> get file from 'Test Set'
#							  --> get data, features and remove the useless features we find in training
#									these features is stored in file "all_deleted_features.csv"
#							  --> note: the label will be set as label = " "
# output: data features is obvious
#		label --> according to the input para(for_train)
def load_data_for_solve(file_name, for_train = True):
	label = ""
	if for_train:
		# for training, load the data from train directory
		original_features, original_data, original = load_data(file_name)
		original_features, original_data, train_label = extract_target(original_features, original_data)
		label = train_label.copy()
		save_result(label, "train_label_original.csv")
	else:
		# for testing or else, load the data from other place
		original_features, original_data, original = load_data(file_name, data_style = "Test Set")
		# if not for train, we should first delete the features we deleted during the training
		known_deleted_features = np.array(load_result("all_deleted_features.csv"))
		known_deleted_features = known_deleted_features.reshape((known_deleted_features.size, ))
		original_data, original_features, deleted_features = delete_features(original_data, original_features, delete_feas_list = known_deleted_features)
		#print(deleted_features)
	data = original_data.copy()
	features = original_features.copy()
	save_result(data, "extractedTarget_originalData.csv", features)

	return data, features, label

# second step to remove features with too many fissing, fill the miss in the data
#	store some useful information
# input: data features is obvious
#	- label: for train please set this a array
#			 for test, just ignore
#	- for_train: default True --> fill the missing with the average of those have same label
#				 set False 	  --> 
#						- fill_with_experience: default False --> filling missing with 
#																input data
#												set True --> filling missing with 
#															things stored in the file
# output: new_data new_feaures
#		- for_train(True) <-- removing all the features with too many missing
#						  <-- filling the missing
#
def filling_miss(data, features, label = "", for_train = True, fill_with_experience = False):
	delete_fea = []
	missing_num = []
	new_data = data.copy()
	new_features = features.copy()
	if for_train:
		#!start from range(1,...) is because the first line of the feature is the id, useless
		for fea_pos in range(1, len(features)):
			fea_val_cla = feature_value_class(data, fea_pos, label)
			if not fea_val_cla["miss"]._present_num == 0:
				new_data = filling_miss_with_current_set(new_data, fea_pos, fea_val_cla, label, \
													delete_fea, missing_num)
		new_data, new_features, deleted_feas = delete_features(new_data, new_features, delete_fea)
		save_result(np.array(missing_num), "deleted_features_with_too_many_missing.csv", np.array(deleted_feas))
		write_to_deleted_features_area(np.array(deleted_feas))

		save_result(new_data, "data_after_filling_missing.csv", new_features)

	else:
		if fill_with_experience:
			for fea_pos in range(1, len(features)):
				new_data = filling_miss_with_experience(new_data, features, fea_pos)
		else:
			for fea_pos in range(1, len(features)):
				fea_val_cla = feature_value_class(new_data, fea_pos, label)
				if not fea_val_cla["miss"]._present_num == 0:
					new_data = filling_miss_with_current_set(new_data, fea_pos, fea_val_cla, label, \
														delete_fea, missing_num)

	save_result(new_data, "data_after_filling_missing.csv", new_features)
	return new_data, new_features

# this step is just for training!!!
#	aim to remove all these with no discrimination features
#	then store all the information in the file
def remove_no_discrimination(data, features, label):
	try:
		a = label.shape
	except:
		return data, features
	index_entroy = sort_features_with_entroy(data, features, label)
	new_data, new_features, deleted_features, delete_fea_entroy = delete_no_discrimination_features(data, features, index_entroy)
	save_result(np.array(delete_fea_entroy), "deleted_features_with_no_discrimination(entroy).csv", np.array(deleted_features))
	save_result(new_data, "data_after_delete_no_discrimination_features.csv", new_features)
	save_features_info(new_data, new_features, label, "after_delete_features_infos.csv")
	write_to_deleted_features_area(np.array(deleted_features))

	return new_data, new_features


# this function is aim to map str style features into digit
#	these special features contain : city -- > UserInfo_2", "UserInfo_4", "UserInfo_7"
#							province --> UserInfo_6, UserInfo_18
#							phone --> UserInfo_8
#							marry --> UserInfo_21
#							resident --> UserInfo_23
#	normal str style features --> map just as their value

def strStyle_features_to_digit(data, features):
	str_style_features = np.array(load_result("str_features.csv")[0])
	city_features = ["UserInfo_2", "UserInfo_4", "UserInfo_7", "UserInfo_19"]
	privince_features = ["UserInfo_6", "UserInfo_18"]
	phone_features = ["UserInfo_8"]
	marry_features = ["UserInfo_21"]
	resident_features = ["UserInfo_23"]

	digited_special_str_features = list()

	# digit city features
	digit_city_data = digit_city_features(data, features, city_features, use_original_features = True)
	save_result(digit_city_data, "digited_city_data.csv", features)
	digited_special_str_features.extend(city_features)
	# digit province features
	digited_province_data = digit_province_features(digit_city_data, features, privince_features, use_original_features = True)
	save_result(digited_province_data, "digited_province_data.csv", features)
	digited_special_str_features.extend(privince_features)
	# digit phone features
	digited_phone_data = digit_phone_features(digited_province_data, features, phone_features, use_original_features = True)
	save_result(digited_phone_data, "digited_phone_data.csv", features)
	digited_special_str_features.extend(phone_features)
	# digit marrage features
	digited_marrage_data = digit_marry_features(digited_phone_data, features, marry_features, use_original_features = True)
	save_result(digited_marrage_data, "digited_marrage_data.csv", features)
	digited_special_str_features.extend(marry_features)
	# digit resident features
	digited_residence_data = digit_resident_features(digited_marrage_data, features, resident_features, use_original_features = True)
	save_result(digited_residence_data, "digited_residence_data.csv", features)
	digited_special_str_features.extend(resident_features)

	digited_special_features_data = digited_residence_data

	digited_data, features_map_info = map_str_to_digit(digited_special_features_data, features, digited_special_str_features)

	#digited_data = convert_to_numerical(convert_to_digit)
	save_result(digited_data, "after_Str_features_digited_data.csv", features)
	save_result(np.array(features_map_info), "digit_Str_features_map_infos.csv", dir_name = "resultData/features_map")


	return digited_data


if __name__ == '__main__':
	################# step1: load the original data ###################
	# "PPD_Training_Master_GBK_3_1_Training_Set.csv"
	# data, features, label = load_data_for_solve("PPD_Training_Master_GBK_3_1_Training_Set.csv")
	# # print(features.shape)
	# # print(data.shape)
	# # print(label.shape)
	# # print(features)
	# # print(data[:3])
	# # print(label)
	# ################## step2: filling_miss ##################
	# data, features = filling_miss(data, features, label)
	# save_features_info(data, features, label, "after_filling_features_info.csv")
	# ################ for train --> find and remove no discrimination features ##########
	# data, features = remove_no_discrimination(data, features, label)

	# ################ map some special features to digit
	# # contents = load_result("data_after_delete_no_discrimination_features.csv")
	# # features = np.array(contents[0])
	# # data = np.array(contents[1:])
	# # label_lines = np.array(load_result("train_label_original.csv"))
	# # from save_load_result import convert_to_float
	# # label = convert_to_float(label_lines)
	# data = strStyle_features_to_digit(data, features)

	# print(features)
	# print(data[:3])

	############ for test ####################
	# data, features, label = load_data_for_solve("PPD_Master_GBK_2_Test_Set.csv", for_train = False)
	# data, features = filling_miss(data, features, for_train = False)
	# data = strStyle_features_to_digit(data, features)

	# print(features)
	# print(data[:3])
