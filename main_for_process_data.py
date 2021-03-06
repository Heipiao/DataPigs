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
from solve_data import feature_value_class, replace_miss_with_specialV, \
						delete_features, \
						sort_features_with_entroy, delete_no_discrimination_features, \
						get_known_features_index

from map_features_to_digit import digit_city_features, digit_province_features, \
								digit_phone_features, digit_marry_features, \
								digit_resident_features, \
								map_str_to_digit, map_str_to_digit_with_experience, \
								FEATURES_MAP_INFO_FILE_NAME, convert_to_numerical

from save_load_result import save_result, save_features_info, \
							load_result, write_to_deleted_features_area, \
							load_all_deleted_features_during_train
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
		SAVE_DIR = "resultData/"
		# for training, load the data from train directory
		original_features, original_data, original = load_data(file_name)
		original_features, original_data, train_label = extract_target(original_features, original_data)
		label = train_label.copy()
		save_result(label, "train_label_original.csv", dir_name = SAVE_DIR)
	else:
		SAVE_DIR = "resultData/test/"
		# for testing or else, load the data from other place
		original_features, original_data, original = load_data(file_name, data_style = "Test Set")
		#print(deleted_features)
	data = original_data.copy()
	features = original_features.copy()
	save_result(data, "withoutLabel_originalData.csv", features, dir_name = SAVE_DIR)

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
def replace_miss(data, features, label = "", for_train = True):
	delete_fea_index = []
	missing_num = []
	new_data = data.copy()
	new_features = features.copy()

	if for_train:
		SAVE_DIR = "resultData"
	else:
		SAVE_DIR = "resultData/test/"
	#!start from range(1,...) is because the first line of the feature is the id, useless
	for fea_pos in range(1, len(features)):
		fea_val_cla = feature_value_class(data, fea_pos, label)
		if not fea_val_cla[-1]._present_num == 0:
			new_data = replace_miss_with_specialV(new_data, fea_pos, fea_val_cla, label, \
												delete_fea_index, missing_num)
	if for_train:
		new_data, new_features, deleted_feas = delete_features(new_data, new_features, \
															delete_fea_pos = delete_fea_index)

		save_result(new_data, "data_after_delete_too_many_missing_features.csv", new_features)

		save_result(np.array(missing_num), "deleted_features_with_too_many_missing.csv", \
					np.array(deleted_feas), dir_name = SAVE_DIR)


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
	save_result(np.array(delete_fea_entroy), "deleted_features_with_no_discrimination(entroy).csv", \
				np.array(deleted_features))
	save_result(new_data, "data_after_delete_no_discrimination_features.csv", \
				new_features)
	save_features_info(new_data, new_features, label, "infos_after_delete_features.csv")
	#write_to_deleted_features_area(np.array(deleted_features))

	return new_data, new_features


# this function is aim to map str style features into digit
#	these special features contain : city -- > UserInfo_2", "UserInfo_4", "UserInfo_7"
#							province --> UserInfo_7, UserInfo_19
#							phone --> UserInfo_9
#							marry --> UserInfo_22
#							resident --> UserInfo_24
#	normal str style features --> map just as their value

def strStyle_features_to_digit(data, features, for_train = True, use_experience = False, save_dir = "resultData/"):

	city_features = ["UserInfo_2", "UserInfo_4", "UserInfo_8", "UserInfo_20"]
	privince_features = ["UserInfo_7", "UserInfo_19"]
	phone_features = ["UserInfo_9"]
	marry_features = ["UserInfo_22"]
	resident_features = ["UserInfo_24"]

	# features_with_simply_value = ["WeblogInfo_2", "WeblogInfo_5", "WeblogInfo_8",
	# 								"UserInfo_10", "UserInfo_18", "WeblogInfo_24",
	# 								"WeblogInfo_27", "WeblogInfo_30"]
	# it is no need to map the other str style features, 
	#	only when the other features contain special characters
	# --> the list below contain all those features
	contain_special_features = ["UserInfo_23", "Education_Info2", "Education_Info3", \
								"Education_Info4", "Education_Info6", "Education_Info7", \
								"WeblogInfo_19", "WeblogInfo_20", "Education_Info8", \
								"WeblogInfo_21", "ListingInfo"]


	digited_special_str_features = list()

	# digit city features
	digit_city_data = digit_city_features(data, features, city_features, use_original_features = True)
	#save_result(digit_city_data, "digited_city_data.csv", features)
	digited_special_str_features.extend(city_features)
	# digit province features
	digited_province_data = digit_province_features(digit_city_data, features, privince_features, \
													use_original_features = True)
	#save_result(digited_province_data, "digited_province_data.csv", features)
	digited_special_str_features.extend(privince_features)
	# digit phone features
	digited_phone_data = digit_phone_features(digited_province_data, features, phone_features, \
												use_original_features = True)
	#save_result(digited_phone_data, "digited_phone_data.csv", features)
	digited_special_str_features.extend(phone_features)
	# digit marrage features
	digited_marrage_data = digit_marry_features(digited_phone_data, features, marry_features, \
												use_original_features = True)
	#save_result(digited_marrage_data, "digited_marrage_data.csv", features)
	digited_special_str_features.extend(marry_features)
	# digit resident features
	digited_residence_data = digit_resident_features(digited_marrage_data, features, resident_features, \
													use_original_features = True)
	save_result(digited_residence_data, "data_when_digited_residence.csv", features)
	digited_special_str_features.extend(resident_features)

	digited_special_features_data = digited_residence_data
	
	if not for_train:
		use_experience = True
		save_dir = "resultData/test"
	# if this map is just or train, which means we do not have experience map style
	if not use_experience:
		digited_data, features_map_info = map_str_to_digit(digited_special_features_data, \
															features, digited_special_str_features, \
															contain_special_features)
		save_result(features_map_info, \
							FEATURES_MAP_INFO_FILE_NAME, \
							dir_name = "resultData/features_map")

	else:
		digited_data = map_str_to_digit_with_experience(digited_special_features_data, \
													features, digited_special_str_features, \
													contain_special_features)

	#digited_data = convert_to_numerical(convert_to_digit)
	save_result(digited_data, "data_after_Str_features_digited.csv", features, dir_name = save_dir)


	return digited_special_features_data

################## we create two method to fill the missed value ########
#### instead this one is not that good

from solve_data import fill_the_missing
def fill_all_missing(data, features, label = None):
	fixed_str_features = np.array(load_result("str_features.csv"))[0]
	indexs = get_known_features_index(features, fixed_str_features)
	#!start from range(1,...) is because the first line of the feature is the id, useless
	for fea_pos in range(1, len(features)):
		fea_val_cla = feature_value_class(data, fea_pos, label, indexs)
		if not fea_val_cla[-1]._present_num == 0:
			if fea_pos == 5:
				print(fea_val_cla)
			data = fill_the_missing(data, fea_pos, fea_val_cla, label)
	#write_to_deleted_features_area(np.array(deleted_feas))
	return data, features

#### so we fill the missed with a sample method ####
