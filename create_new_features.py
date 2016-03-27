#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-15 17:26:35
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

from statistic_data_info import missed_instances_info
from save_load_result import save_result, load_result, load_all_deleted_features_during_train
from map_features_to_digit import convert_to_numerical
from solve_data import get_known_features_index
from solve_data import delete_features
import numpy as np
from functools import reduce

from collections import OrderedDict
## all the new features is based on "original that just remove the missing features"
#### Ie. data_after_delete_too_many_missing_features.csv


def find_featuers_index(features_name, features):
	fea_index = list()
	for fea_pos in range(1, len(features)):
		if features_name in features[fea_pos]:
			fea_index.append(fea_pos)
	return fea_index

def count_missed_create_new_feature(data, features, key_word):
	fea_indexs = find_featuers_index(key_word, features)

	feature_name = key_word + "_missed_count"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.ones((data.shape[0], 1))

	for user in range(data.shape[0]):
		user_miss_count = 0
		for fea_pos in fea_indexs:
			if data[user, fea_pos] == "-1":
				user_miss_count += 1
		feature_data[user, 0] = user_miss_count
	new_data = np.concatenate((data, feature_data), axis = 1)

	#print(set(list(feature_data.reshape((feature_data.size, )))))

	return new_data, new_features


def new_UserInfo_miss_count(data, features):
	key_word = "UserInfo"
	new_data, new_features = count_missed_create_new_feature(data, features, key_word)
	print("count user info missed")
	return new_data, new_features

def new_UserInfo_2_level(data, features):
	from map_features_to_digit import digit_city_features
	city_features = ["UserInfo_2"]

	digited_city_data = digit_city_features(data, features, city_features)

	new_add_feature = np.array(["UserInfo_2_city_level"])
	new_features = np.concatenate((features, new_add_feature))
	
	feature_data = digited_city_data[:, np.where(features == city_features)[0][0]]
	feature_data = feature_data.reshape((feature_data.size, 1))
	#print(feature_data)
	#print("sdf", feature_data.shape)

	new_data = np.concatenate((data, feature_data), axis = 1)

	print("UserInfo_2_city_level" + " solved")
	return new_data, new_features


def new_UserInfo_7_level(data, features):
	from map_features_to_digit import digit_province_features
	province_features = ["UserInfo_7"]

	digited_city_data = digit_province_features(data, features, province_features)

	new_add_feature = np.array(["UserInfo_7_province_level"])
	new_features = np.concatenate((features, new_add_feature))
	
	feature_data = digited_city_data[:, np.where(features == province_features)[0][0]]
	feature_data = feature_data.reshape((feature_data.size, 1))
	#print(feature_data)
	#print("sdf", feature_data.shape)

	new_data = np.concatenate((data, feature_data), axis = 1)

	print("UserInfo_7_province_level" + " solved")
	return new_data, new_features

#### solve features UserInfo_7 UserInfo_8 UserInfo_9 #####

def new_UserInfo_789(data, features):
	solved_features = ["UserInfo_7", "UserInfo_8", "UserInfo_9"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "UserInfo_7_8_9_wrong_phone_address"
	new_add_feature = np.array([feature_name])

	feature_data = np.zeros((len(data), 1))
	new_features = np.concatenate((features, new_add_feature))

	for user in range(data.shape[0]):
		if str(-1) in list(data[user, fea_indexs]):
			feature_data[user, 0] = 1

	new_data = np.concatenate((data, feature_data), axis = 1)

	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = solved_features)
	print(deleted)
	return new_data, new_features

def new_UserInfo_11_12_13(data, features):
	solved_features = ["UserInfo_11", "UserInfo_12", "UserInfo_13"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "UserInfo_11_12_13_info"
	new_add_feature = np.array([feature_name])

	feature_data = np.zeros((len(data), 1))
	new_features = np.concatenate((features, new_add_feature))

	for user in range(data.shape[0]):
		combine_data = list(map(int, list(data[user, fea_indexs])))
		add = reduce(lambda x, y: x * 2 + y, combine_data)
		if add > 0:
			feature_data[user, 0] = add
		else:
			feature_data[user, 0] = 0

	new_data = np.concatenate((data, feature_data), axis = 1)

	print("extract from UserInfo 11 12 13")
	return new_data, new_features



def new_UserInfo_differ(data, features, key_features, feature_name, deleted_all = True):
	
	fea_indexs = get_known_features_index(features, key_features)

	
	new_add_feature = np.array([feature_name])

	feature_data = np.zeros((len(data), 1))
	new_features = np.concatenate((features, new_add_feature))

	for user in range(data.shape[0]):
		if not len(set(list(data[user, fea_indexs]))) == 1:
			feature_data[user, 0] = 1

	new_data = np.concatenate((data, feature_data), axis = 1)

	delete_feas = key_features[1:]
	if deleted_all:
		delete_feas = key_features

	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = delete_feas)
	print(deleted)
	return new_data, new_features

# here we bin the age to some bins
def new_UserInfo_18(data, features):
	solved_features = ["UserInfo_18"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "UserInfo_18_bined"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))
	for user in range(data.shape[0]):
		user_age = data[user, fea_indexs]
		if user_age < "22":
			feature_data[user, 0] = 0
		elif user_age < "30":
			feature_data[user, 0] = 1
		elif user_age < "40":
			feature_data[user, 0] = 2
		elif user_age < "50":
			feature_data[user, 0] = 3
		else:
			feature_data[user, 0] = 4

	new_data = np.concatenate((data, feature_data), axis = 1)

	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = solved_features)
	print(deleted)
	return new_data, new_features


def new_UserInfo_19_20(data, features):
	solved_features = ["UserInfo_19", "UserInfo_20"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "UserInfo_19_20_wrong_province_city"
	new_add_feature = np.array([feature_name])

	feature_data = np.zeros((len(data), 1))
	new_features = np.concatenate((features, new_add_feature))

	for user in range(data.shape[0]):
		if str(-1) in list(data[user, fea_indexs]):
			feature_data[user, 0] = 1

	new_data = np.concatenate((data, feature_data), axis = 1)

	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = solved_features)
	print(deleted)
	return new_data, new_features

from map_features_to_digit import digit_marry_features
def new_UserInfo_22_marrage(data, features):
	marrage_fea = ["UserInfo_22"]
	digited_marrage_data = digit_marry_features(data, features, marrage_fea)

	new_add_feature = np.array(["UserInfo_22_marrage_info"])
	new_features = np.concatenate((features, new_add_feature))
	
	feature_data = digited_marrage_data[:, np.where(features == marrage_fea)[0][0]]
	feature_data = feature_data.reshape((feature_data.size, 1))
	#print(feature_data)
	#print("sdf", feature_data.shape)

	new_data = np.concatenate((data, feature_data), axis = 1)

	print("UserInfo_22_marrage_level" + " solved")
	return new_data, new_features

from map_features_to_digit import digit_education_level_features
def new_UserInfo_23_education_level(data, features):
	education_level_feature = ["UserInfo_23"]
	digited_EL_data = digit_education_level_features(data, features, education_level_feature)

	new_add_feature = np.array(["UserInfo_23_education_level"])
	new_features = np.concatenate((features, new_add_feature))
	
	feature_data = digited_EL_data[:, np.where(features == education_level_feature)[0][0]]
	feature_data = feature_data.reshape((feature_data.size, 1))
	#print(feature_data)
	#print("sdf", feature_data.shape)

	new_data = np.concatenate((data, feature_data), axis = 1)

	print("UserInfo_23_education_level" + " solved")
	return new_data, new_features

from map_features_to_digit import digit_resident_features
def new_UserInfo_24_resident_level(data, features):
	resident_detail_level = ["UserInfo_24"]
	digited_residence_data = digit_resident_features(data, features, resident_detail_level)

	new_add_feature = np.array(["UserInfo_24_resident_detail_level"])
	new_features = np.concatenate((features, new_add_feature))
	
	feature_data = digited_residence_data[:, np.where(features == resident_detail_level)[0][0]]
	feature_data = feature_data.reshape((feature_data.size, 1))
	#print(feature_data)
	#print("sdf", feature_data.shape)

	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = resident_detail_level)

	print("UserInfo_24_resident_detail_level" + " solved")
	print(deleted)
	return new_data, new_features



# def new_UserInfo_22_23_combine1(data, features):
# 	key_features = ["UserInfo_22", "UserInfo_23"]
# 	print("combine1")
# 	fea_indexs = get_known_features_index(features, key_features)
# 	feature_name = "UserInfo_combine_by_label_22_23"
# 	new_add_feature = np.array([feature_name])
# 	new_features = np.concatenate((features, new_add_feature))

# 	##### map rules ####
# 	map_to_zero = [['离婚', 'AB'], ['已婚', 'Z'], ['离婚', 'P'], ['未婚', 'X'], ['-1', 'AI'], 
# 					['未婚', 'AJ'], ['未婚', 'AP'], ['离婚', 'G'], ['再婚', 'M'], ['未婚', 'Q'], 
# 					['已婚', 'K'], ['离婚', 'AC'], ['已婚', '-1'], ['已婚', 'AH'], ['未婚', 'AH'], 
# 					['未婚', 'R'], ['-1', 'AH'], ['再婚', 'H'], ['-1', 'R'], ['再婚', 'O'], 
# 					['已婚', 'P'], ['已婚', '大学本科（简称“大学'], ['离婚', '大学本科（简称“大学'], 
# 					['已婚', 'AJ'], ['初婚', 'G'], ['-1', 'AC'], ['-1', 'P'], ['未婚', 'Y'], 
# 					['未婚', 'AD'], ['离婚', 'H'], ['已婚', 'AF'], ['-1', 'K'], ['未婚', 'K'], 
# 					['已婚', 'Y'], ['未婚', 'W'], ['-1', '专科毕业'], ['已婚', 'AE'], ['未婚', 'AE'], 
# 					['已婚', 'AB'], ['离婚', 'K'], ['离婚', '-1'], ['-1', '大学本科（简称“大学'], 
# 					['再婚', 'G'], ['离婚', '专科毕业'], ['离婚', 'M'], ['-1', 'Y'], ['已婚', 'AL'], 
# 					['-1', 'M'], ['未婚', '专科毕业'], ['已婚', 'M']]

# 	map_to_one = [['未婚', 'P'], ['-1', 'O'], ['离婚', 'O'], ['已婚', 'AK'], ['离婚', 'AH'], 
# 					['已婚', 'AC'], ['已婚', 'AI'], ['未婚', '大学本科（简称“大学'], ['未婚', 'AI'], 
# 					['-1', 'H'], ['-1', 'AK'], ['未婚', 'AC'], ['未婚', '-1'], ['未婚', 'M'], 
# 					['已婚', '专科毕业'], ['-1', 'G'], ['未婚', 'H'], ['已婚', 'H'], ['-1', 'AB'], 
# 					['未婚', 'AK'], ['已婚', 'O'], ['已婚', 'G']]

# 	map_to_two = [['未婚', 'O'], ['未婚', 'AB']]
# 	map_to_three = [['未婚', 'G']]
# 	map_to_four = [['D', 'D']]
# 	map_to_five = [['-1', '-1']]

# 	none_finded_combine = OrderedDict()

# 	feature_data = np.ones((len(data), 1))
# 	for user in range(data.shape[0]):
# 		EI_22_23 = list(data[user, fea_indexs])
# 		if EI_22_23 in map_to_zero:
# 			feature_data[user, 0] = 0
# 		elif EI_22_23 in map_to_one:
# 			feature_data[user, 0] = 1
# 		elif EI_22_23 in map_to_two:
# 			feature_data[user, 0] = 2
# 		elif EI_22_23 in map_to_three:
# 			feature_data[user, 0] = 3
# 		elif EI_22_23 in map_to_four:
# 			feature_data[user, 0] = 4
# 		elif EI_22_23 in map_to_five:
# 			feature_data[user, 0] = 5
# 		else:
# 			print("error!!!!")
# 			print(EI_22_23)
# 	# 		if EI_22_23 not in none_finded_combine.keys():
# 	# 			none_finded_combine[EI_22_23] = list()
# 	# 		none_finded_combine[EI_22_23].append(user)

# 	# for EI_combine, users in none_finded_combine.items():
# 	# 	if len(users)

# 	new_data = np.concatenate((data, feature_data), axis = 1)
# 	return new_data, new_features

def new_UserInfo_22_23_combine2(data, features):
	key_features = ["UserInfo_22", "UserInfo_23"]
	print("combine2")
	fea_indexs = get_known_features_index(features, key_features)
	feature_name = "UserInfo_combine2_by_present_22_23"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	##### map rules ####
	map_to_zero = [['未婚', 'AE'], ['已婚', 'AH'], ['再婚', 'M'], ['未婚', 'X'], ['未婚', 'AJ'], 
					['-1', 'AC'], ['离婚', 'K'], ['已婚', 'AF'], ['未婚', 'AP'], ['再婚', 'G'], 
					['未婚', 'R'], ['已婚', 'AL'], ['离婚', '专科毕业'], ['离婚', 'G'], ['离婚', 'AC'], 
					['未婚', 'AD'], ['-1', 'M'], ['离婚', 'AB'], ['已婚', 'AJ'], ['-1', 'R'], 
					['已婚', 'Y'], ['离婚', 'H'], ['未婚', 'Q'], ['离婚', 'P'], ['已婚', 'Z'], 
					['初婚', 'G'], ['-1', 'K'], ['再婚', 'O'], ['-1', 'AI'], ['离婚', '-1'], 
					['已婚', '-1'], ['再婚', 'H'], ['未婚', 'AH'], ['离婚', '大学本科（简称“大学'], 
					['离婚', 'M'], ['-1', 'P'], ['已婚', 'AE'], ['-1', '专科毕业'], ['-1', 'AH'], 
					['已婚', 'P'], ['已婚', 'AI'], ['离婚', 'AH'], ['离婚', 'O'], ['已婚', 'AC'], 
					['-1', 'H'], ['未婚', 'AC'], ['-1', 'AK']]

	map_to_one = [['已婚', 'K'], ['未婚', 'K'], ['未婚', 'W'], ['-1', '大学本科（简称“大学'], ['已婚', '专科毕业']]

	map_to_two = [['未婚', 'Y'], ['已婚', 'AB'], ['未婚', '专科毕业'], ['已婚', '大学本科（简称“大学'], 
					['已婚', 'M'], ['-1', 'Y'], ['未婚', 'P'], ['-1', 'O'], ['已婚', 'AK'], 
					['未婚', 'AI'], ['未婚', 'M'], ['未婚', '-1'], ['-1', 'G'], ['未婚', 'H'], 
					['已婚', 'H'], ['-1', 'AB'], ['未婚', 'AK'], ['已婚', 'O']]
	map_to_three = [['未婚', '大学本科（简称“大学'], ['已婚', 'G'], ['未婚', 'O'], ['未婚', 'AB'], ['未婚', 'G']]
	map_to_four = [['D', 'D']]
	map_to_five = [['-1', '-1']]
	none_finded_combine = OrderedDict()
	feature_data = np.ones((len(data), 1))
	for user in range(data.shape[0]):
		EI_22_23 = list(data[user, fea_indexs])
		if EI_22_23 in map_to_zero:
			feature_data[user, 0] = 0
		elif EI_22_23 in map_to_one:
			feature_data[user, 0] = 1
		elif EI_22_23 in map_to_two:
			feature_data[user, 0] = 2
		elif EI_22_23 in map_to_three:
			feature_data[user, 0] = 3
		elif EI_22_23 in map_to_four:
			feature_data[user, 0] = 4
		elif EI_22_23 in map_to_five:
			feature_data[user, 0] = 5
		else:
			EI_22_23_str = reduce(lambda x, y: x + "_" + y, EI_22_23)
			if EI_22_23_str not in none_finded_combine.keys():
				none_finded_combine[EI_22_23_str] = list()
			none_finded_combine[EI_22_23_str].append(user)

	for EI_combine, users in none_finded_combine.items():
		EI_combine = EI_combine.split("_")
		if EI_combine[0] == "-1" and EI_combine[1] == "-1":
			feature_data[users, 0] = 5
		if len(users) < 10:
			feature_data[users, 0] = 0
		elif len(users) < 20:
			feature_data[users, 0] = 1
		elif len(users) < 100:
			feature_data[users, 0] = 2
		elif len(users) < 1000:
			feature_data[users, 0] = 3
		else:
			feature_data[users, 0] = 4

	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = key_features)
	print(deleted)
	return new_data, new_features
# note: we just extract features from the user info 22 23 24,
# 		so we can not delete them here
#def new_UserInfo_22_23_24(data, features):


######################## create new features from education ##############
def new_EI_2(data, features):
	solved_features = ["Education_Info2"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "Education_Info2_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "B":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "U":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "AN":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "AQ":
			feature_data[user, 0] = 3
		elif data[user, fea_indexs[0]] == "A":
			feature_data[user, 0] = 4
		elif data[user, fea_indexs[0]] == "AM":
			feature_data[user, 0] = 5
		elif data[user, fea_indexs[0]] == "E":
			feature_data[user, 0] = 6
		else:
			print("error in Education_Info2")

	
	new_data = np.concatenate((data, feature_data), axis = 1)

	print("Education_Info2 solved")
	return new_data, new_features

def new_EI_4(data, features):
	solved_features = ["Education_Info4"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "Education_Info4_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "AE":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "AR":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "V":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "F":
			feature_data[user, 0] = 3
		elif data[user, fea_indexs[0]] == "T":
			feature_data[user, 0] = 4
		elif data[user, fea_indexs[0]] == "E":
			feature_data[user, 0] = 5
		else:
			print("error in Education_Info4")

	
	new_data = np.concatenate((data, feature_data), axis = 1)

	print("Education_Info4 solved")
	return new_data, new_features


## Education_Info1 shows another information, we can not remove it here
def new_EI_1_2_3_4(data, features):
	key_features = ["Education_Info1", "Education_Info2", "Education_Info3", "Education_Info4"]
	fea_indexs = get_known_features_index(features, key_features)
	feature_name = "combine_EI_1_2_3_4"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	##### map rules ####
	map_to_zero = [["1", "AQ", "毕业", "T"], ["1", "A", "毕业", "V"], ["1", "AN", "结业", "T"], 
					["1", "AM", "结业", "T"], ["1", "B", "毕业", "AE"], ["1", "A", "结业", "T"],
					["1", "A", "毕业", "AR"]]
	map_to_one = [["1", "U", "毕业", "AE"], ["1", "AM", "毕业", "AR"], ["1", "AM", "毕业", "V"], 
					["1", "AQ", "毕业", "F"], ["1", "A", "毕业", "F"], ["1", "AN", "毕业", "T"],
					["1", "AQ", "毕业", "V"]]
	map_to_two = [["1", "AM", "毕业", "T"], ["1", "A", "毕业", "T"], ["1", "AM", "毕业", "F"]]
	map_to_three = [["0", "E", "E", "E"]]

	feature_data = np.ones((len(data), 1))
	for user in range(data.shape[0]):
		EI_1_2_3_4 = list(data[user, fea_indexs])
		if EI_1_2_3_4 in map_to_zero:
			feature_data[user, 0] = 0
		elif EI_1_2_3_4 in map_to_one:
			feature_data[user, 0] = 1
		elif EI_1_2_3_4 in map_to_two:
			feature_data[user, 0] = 2
		elif EI_1_2_3_4 in map_to_three:
			feature_data[user, 0] = 3
		else:
			print("error!!!!")
	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = key_features[1:])
	print(deleted)
	return new_data, new_features


def new_EI_6(data, features):
	solved_features = ["Education_Info6"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "Education_Info6_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "B":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "U":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "AQ":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "AM":
			feature_data[user, 0] = 3
		elif data[user, fea_indexs[0]] == "A":
			feature_data[user, 0] = 4
		elif data[user, fea_indexs[0]] == "E":
			feature_data[user, 0] = 5
		else:
			print("error in Education_Info6")

	
	new_data = np.concatenate((data, feature_data), axis = 1)

	print("Education_Info6 solved")
	return new_data, new_features

def new_EI_8(data, features):
	solved_features = ["Education_Info8"]
	fea_indexs = get_known_features_index(features, solved_features)

	feature_name = "Education_Info8_info_(cat)"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	feature_data = np.zeros((len(data), 1))

	for user in range(data.shape[0]):
		if data[user, fea_indexs[0]] == "V" or data[user, fea_indexs[0]] == "AE":
			feature_data[user, 0] = 0
		elif data[user, fea_indexs[0]] == "80":
			feature_data[user, 0] = 1
		elif data[user, fea_indexs[0]] == "F":
			feature_data[user, 0] = 2
		elif data[user, fea_indexs[0]] == "T":
			feature_data[user, 0] = 3
		elif data[user, fea_indexs[0]] == "-1":
			feature_data[user, 0] = 4
		elif data[user, fea_indexs[0]] == "E":
			feature_data[user, 0] = 5
		else:
			print("error in Education_Info8")

	
	new_data = np.concatenate((data, feature_data), axis = 1)

	print("Education_Info8 solved")
	return new_data, new_features

## Education_Info5 shows another information, we can not remove it here
def new_EI_5_6_7_8(data, features):
	key_features = ["Education_Info5", "Education_Info6", "Education_Info7", "Education_Info8"]
	fea_indexs = get_known_features_index(features, key_features)
	feature_name = "combine_EI_5_6_7_8"
	new_add_feature = np.array([feature_name])
	new_features = np.concatenate((features, new_add_feature))

	##### map rules ####
	map_to_zero = [["1", "AQ", "-1", "T"], ["1", "AQ", "-1", "80"], ["1", "U", "-1", "-1"], 
					["1", "AQ", "-1", "-1"], ["1", "B", "-1", "-1"], ["1", "A", "-1", "-1"],
					["1", "AM", "-1", "80"], ["1", "A", "-1", "F"], ["1", "B", "-1", "AE"], 
					["1", "U", "-1", "AE"], ["1", "AQ", "-1", "V"], ["1", "AM", "-1", "V"]]

	map_to_one = [["1", "A", "-1", "T"], ["1", "AQ", "-1", "F"], ["1", "AM", "-1", "-1"], 
					["1", "AM", "-1", "-1"], ["1", "AM", "-1", "F"], ["1", "AM", "-1", "T"]]
	map_to_two = [["0", "E", "E", "E"]]

	feature_data = np.ones((len(data), 1))
	for user in range(data.shape[0]):
		EI_5_6_7_8 = list(data[user, fea_indexs])
		if EI_5_6_7_8 in map_to_zero:
			feature_data[user, 0] = 0
		elif EI_5_6_7_8 in map_to_one:
			feature_data[user, 0] = 1
		elif EI_5_6_7_8 in map_to_two:
			feature_data[user, 0] = 2
		else:
			print("error!!!!")
	new_data = np.concatenate((data, feature_data), axis = 1)
	new_data, new_features, deleted = delete_features(new_data, new_features, \
										delete_feas_list = key_features[1:])
	print(deleted)
	return new_data, new_features 




def solve_user_info_package(data, features, saved_dir = "resultData"):
	data, features = count_missed_create_new_feature(data, features, "UserInfo")

	# ################### solve the education info ####################
	data, features = new_EI_2(data, features)
	data, features = new_EI_4(data, features)
	data, features = new_EI_1_2_3_4(data, features)
#	save_result(data, "data_after_combine_EI1234.csv", features, dir_name = saved_dir)
	data, features = new_EI_6(data, features)
	data, features = new_EI_8(data, features)
	data, features = new_EI_5_6_7_8(data, features)
#	save_result(data, "data_after_combine_EI5678.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_miss_count(data, features)
#	save_result(data, "data_after_count_UserInfo_miss.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_2_level(data, features)
#	save_result(data, "data_after_solved_UserInfo2_level.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_7_level(data, features)
#	save_result(data, "data_after_solved_UserInfo7_level.csv", features, dir_name = saved_dir)

	key_features = ["UserInfo_2", "UserInfo_4"]
	feature_name = "UserInfo_2_4_wrong_correspond_city)"
	data, features = new_UserInfo_differ(data, features, key_features, feature_name)
#	save_result(data, "data_after_solved_UserInfo2_4.csv", features, dir_name = saved_dir)

	key_features = ["UserInfo_5", "UserInfo_6"]
	feature_name = "UserInfo_5_6_differ"
	data, features = new_UserInfo_differ(data, features, key_features, feature_name, deleted_all = False)
#	save_result(data, "data_after_solved_UserInfo5_6.csv", features, dir_name = saved_dir)


	data, features = new_UserInfo_789(data, features)
#	save_result(data, "data_after_solved_UserInfo789.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_11_12_13(data, features)
	key_features = ["UserInfo_11", "UserInfo_12", "UserInfo_13"]
	feature_name = "UserInfo_11_12_13_is_miss"
	data, features = new_UserInfo_differ(data, features, key_features, feature_name)
#	save_result(data, "data_after_solved_UserInfo11_12_13.csv", features, dir_name = saved_dir)

	key_features = ["UserInfo_14", "UserInfo_15"]
	feature_name = "UserInfo_14_15_differ"
	data, features = new_UserInfo_differ(data, features, key_features, feature_name, deleted_all = False)
#	save_result(data, "data_after_solved_UserInfo14_15.csv", features, dir_name = saved_dir)

	key_features = ["UserInfo_16", "UserInfo_17"]
	feature_name = "UserInfo_16_17_differ"
	data, features = new_UserInfo_differ(data, features, key_features, feature_name, deleted_all = False)
#	save_result(data, "data_after_solved_UserInfo16_17.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_18(data, features)
#	save_result(data, "data_after_solved_UserInfo18.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_19_20(data, features)
#	save_result(data, "data_after_solved_UserInfo19_20.csv", features, dir_name = saved_dir)
	
	data, features = new_UserInfo_22_marrage(data, features)
#	save_result(data, "data_after_solved_UserInfo22.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_23_education_level(data, features)
#	save_result(data, "data_after_solved_UserInfo23.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_24_resident_level(data, features)
#	save_result(data, "data_after_solved_UserInfo24.csv", features, dir_name = saved_dir)

	#data, features = new_UserInfo_22_23_combine1(data, features)
#	save_result(data, "data_after_solved1_UserInfo22_23.csv", features, dir_name = saved_dir)

	data, features = new_UserInfo_22_23_combine2(data, features)
	save_result(data, "data_after_solved_user_info.csv", features, dir_name = saved_dir)

	return data, features


if __name__ == '__main__':

	contents = load_result("data_after_delete_too_many_missing_features.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	deleted_features_in_train = load_all_deleted_features_during_train(deleted_features_file_label = "deleted_")
	data, features, deleted = delete_features(data, features, delete_feas_list = deleted_features_in_train)

	data, features = solve_user_info_package(data, features)

	from create_features_from_weblog import solve_weblog_info_package
	data, features = solve_weblog_info_package(data, features)
