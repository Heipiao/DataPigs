#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-29 15:22:21
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: used the features to digit value for easy to solve in the future

from save_load_result import save_result, load_result
from solve_data import feature_value_class, FeatureInData, convert_to_digit, \
						delete_features
import numpy as np
from collections import OrderedDict
# replace the features with the original_data
# input:
#	data: the data you want to replaced
#	replace_features: a list contain the features you want to replace
def replace_with_original(data, features, replace_features):
	original_contents = load_result("extractedTarget_originalData.csv")
	original_features = np.array(original_contents[0])
	original_data = np.array(original_contents[1:])

	for fea in replace_features:
		try:
			original_index = np.where(original_features == fea)[0][0]
			index = np.where(features == fea)[0][0]	
			data[:, index] = original_data[:, original_index]
		except:
			print(str(fea) + "may not existed")
	return data
# so digit the city name to value 
#	the basis idea is according to the cei record of each cities:
#		record >= 80 		--> 0
#		75 <= record < 80 	--> 1
#		70 <= record < 75 	--> 2
#		record < 70 		--> 3
#		no cei info 		--> 4
BEST_CITY = 0
BETTER_CITY = 1
GREAT_CITY = 2
NORMAL_CITY = 3
BAD_CITY = 4
NO_INFO = -10
def create_city_map_basis(cei_data, cei_features):
	city_index = np.where( cei_features == "城市名称")[0][0]
	score_index = np.where( cei_features == "分值")[0][0]

	city_map_basis = dict()
	city_map_basis["noMatch"] = BAD_CITY
	city_map_basis["不详"] = NO_INFO
	for i in range(len(cei_data)):
		city_cei_score = float(cei_data[i, score_index])

		if city_cei_score >= 80:
			city_map_basis[cei_data[i, city_index]] = BEST_CITY
		elif city_cei_score >= 75:
			city_map_basis[cei_data[i, city_index]] = BETTER_CITY
		elif city_cei_score >= 70:
			city_map_basis[cei_data[i, city_index]] = GREAT_CITY
		else:
			city_map_basis[cei_data[i, city_index]] = NORMAL_CITY

	return city_map_basis

# create the map basis for province
#	the basis idea is according to the income of each privince`s person:
#		income >= 30000 		--> 0
#		25000 <= income < 30000		--> 1
#		23000 <= income < 25000 		--> 2
#		income < 23000 			--> 3
BEST_PROVINCE = 0
BETTER_PROVINCE = 1
GREAT_PROVINCE = 2
NORMAL_PROVINCE = 3
BAD_PROVINCE = 4
NO_INFO = -10
def create_province_map_basis(provs_data, provs_features):
	province_index = np.where( provs_features == "省份名")[0][0]
	income_index = np.where( provs_features == "可支配收入")[0][0]

	province_map_basis = dict()
	province_map_basis["noMatch"] = BAD_PROVINCE
	province_map_basis["不详 "] = NO_INFO
	for i in range(len(provs_data)):
		province_person_income = float(provs_data[i, income_index])

		if province_person_income >= 30000:
			province_map_basis[provs_data[i, province_index]] = BEST_PROVINCE
		elif province_person_income >= 25000:
			province_map_basis[provs_data[i, province_index]] = BETTER_PROVINCE
		elif province_person_income >= 23000:
			province_map_basis[provs_data[i, province_index]] = GREAT_PROVINCE
		else:
			province_map_basis[provs_data[i, province_index]] = NORMAL_PROVINCE

	return province_map_basis

# here we create the map basis for phone
#	according to:
#	if has info: 1 中国移动 中国电信 中国联通
#	if it is 不详: -10
def create_phone_map_basis():
	phone_map_basis = dict()
	phone_map_basis["中国移动 "] = 1
	phone_map_basis["中国电信 "] = 1
	phone_map_basis["中国联通 "] = 1
	phone_map_basis["不详 "] = -10

	return phone_map_basis
# 未婚 初婚 已婚 离婚 再婚
def create_marrage_map_basis():
	marrage_map_basis = dict()
	marrage_map_basis["未婚 "] = 0
	marrage_map_basis["初婚 "] = 1
	marrage_map_basis["已婚 "] = 2
	marrage_map_basis["离婚 "] = 3
	marrage_map_basis["再婚 "] = 4
	marrage_map_basis["不详 "] = -10
	marrage_map_basis["D "] = 5
	return marrage_map_basis

######## 
## I believe that the UserInfo_23 may be the residence for a person
## the more the residence is specific, the better
# "省 州 市 县 区 旗 镇 乡 庄 村 路 街 巷 号码"
BEST_SPECIFIC = 0

def create_residence_map_basis():
	residence_map_basis = dict()
	residence_map_basis["市"] = 0
	residence_map_basis["州"] = 0
	residence_map_basis["县"] = 1
	residence_map_basis["区"] = 1
	residence_map_basis["旗"] = 1 
	residence_map_basis["镇"] = 2
	residence_map_basis["乡"] = 2
	residence_map_basis["庄"] = 2
	residence_map_basis["村"] = 2
	residence_map_basis["路"] = 2
	residence_map_basis["街"] = 2
	residence_map_basis["巷"] = 2
	residence_map_basis["D"] = 4
# 
# map_features should be a list contain feature, like:
#	map_features = ["UserInfo_23"]
def map_residence(data, features, map_features):
	digited_data = data.copy()
	import re
	# 
	pattern_bad = re.compile(r".*市.*|.*自治区.*")
	# 0
	pattern_normal = re.compile(r".*州.*|.*县.*|.*区.*|.*旗.*")
	# 1
	pattern_good1 = re.compile(r".*镇.*|.*市.*.*乡.*|.*市.*.*庄.*|.*村.*")
	pattern_good2 = re.compile(r".*路.*|.*街.*|.*巷.*|.*村.*")
	pattern_good3 = re.compile(r".*省.*县.*乡.*|.*省.*县.*村.*|.*省.*县.*镇.*")

	pattern_good5 = re.compile(r".*市.*区.*\d.*")
	# 2
	pattern_great1 = re.compile(r".*市.*区.*\d.*")
	pattern_great = re.compile(r".*\d.*")
	# 3 --> else
	# 4 --> D

	def map_resdent(x):
		#print(x)
		map_result = 3
		if pattern_bad.search(x):
			if pattern_normal.search(x):
				map_result = 0
				if pattern_good1.search(x) or pattern_good2.search(x):
					map_result = 1
					if pattern_great.search(x):
						map_result = 2
		if pattern_good3.search(x):
			map_result = 1
		if pattern_great1.search(x):
			map_result = 2
		if x == "D":
			map_result = 4
		#print(map_result)
		return map_result

	for f in map_features:
		#print(f)
		fea_pos = np.where(features == f)[0][0]
		#print(fea_pos)
		map_result = np.array(list(map(map_resdent, data[:, fea_pos])))
		digited_data[:, fea_pos] = map_result
	return digited_data
			
# as we can see : the city information in features is :
#	User_Info_2, User_Info_4, User_Info_6, User_Info_7, User_Info_18
# if you want to map city features, please use:
#	UserInfo_2", "UserInfo_4", "UserInfo_7"
# if you want to map privince features, please use:
#	UserInfo_6, UserInfo_18 --> map_features = ["UserInfo_6", "UserInfo_18"]
#	---> put this to map_features(list) as a input
def use_map_basis_to_digit(data, features, map_basis, map_features):
	digited_data = data.copy()

	contain_respond_features_index = list()
	for i in range(len(map_features)):
		# we get the first match
		# PLease note: the 'UserInfo_4' is repeated in the features!!!!
		#	instead we just get the first one
		finded_index = np.where( features == map_features[i])[0][0]
		contain_respond_features_index.append(finded_index)

	def map_to(x):
		for k in map_basis.keys():
			if x in k:
				return map_basis[k]
		# not find in keys
		print(x)
		return map_basis["noMatch"]
	# print(map_to(data[0, 2]))
	for fea_index in contain_respond_features_index:
		map_result = np.array(list(map(map_to, data[:, fea_index])))
		# replace the data 
		digited_data[:, fea_index] = map_result

	return digited_data

# convert the string in data into int style
#	I.e --> map the str into value
def map_str_feature_to_value(data, fea_pos, fea_value_sat):
	map_flag = 0
	map_info = dict()
	# make sure the style of map
	for k, v in fea_value_sat.items():
		if isinstance(v, FeatureInData):
			# map the value k of this feature to map_flag
			map_info[k] = map_flag
			map_flag += 1

	for i in range(len(data)):
		data[i, fea_pos] = map_info[data[i, fea_pos]]

	return data, map_info

# map all the str style features into int value
# Please use the sentence below to load the input for this function:
#	# contents = load_result("data_after_delete__no_discrimination_features.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
	# label_lines = np.array(load_result("train_label_original.csv"))
	# from save_load_result import convert_to_float
	# label = convert_to_float(label_lines)
# output:
#	digit_data: the digit data for the origin
#	features_map_info: how to map the str features is stored in this para..
# note: use sentence below to save the result
	# save_result(digit_data, "after_delete_get_digit_data.csv", features)
	# save_result(np.array(features_map_info), "features_map_infos.csv", dir_name = "resultData/features_map")

def map_str_to_digit(data, features, label):

	features_map_info = list()
	for fea_pos in range(1, len(features)):
		map_info = OrderedDict()
		feature_map_info = OrderedDict()
		fea_val_cla = feature_value_class(data, fea_pos, label)
		# if this feature is a string value, just convert it to value
		if fea_val_cla["str_feature"]:
			data, map_info = map_str_feature_to_value(data, fea_pos, fea_val_cla)
			feature_map_info[features[fea_pos]] = map_info
			features_map_info.append([feature_map_info])

	digit_data = convert_to_digit(data)
	return digit_data, features_map_info


if __name__ == '__main__':
	###################### used to digit city features ##################################
	########### features: UserInfo_2", "UserInfo_4", "UserInfo_7"
	# cei_record_content = load_result("2013中国直辖市 省会城市和计划单列市排名榜.csv", dir_name = "material_data")
	# cei_features = np.array(cei_record_content[0])
	# cei_recored_data1 = np.array(cei_record_content[1:])

	# cei_record_content = load_result("2013中国城市商业信用环境指数地级市排名榜.csv", dir_name = "material_data")
	# cei_recored_data2 = np.array(cei_record_content[1:])
	# cei_recored_data = np.concatenate((cei_recored_data1, cei_recored_data2))

	# # print(cei_recored_data)

	# city_map_basis = create_city_map_basis(cei_recored_data, cei_features)
	# for k, v in city_map_basis.items():
	# 	print(k, v)

	


	# contents = load_result("data_after_delete_no_discrimination_features.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
	# # we replace the col with original one, Ie. UserInfo_7
	# replace_features = ["UserInfo_7", "UserInfo_19"]
	# data = replace_with_original(data, features, replace_features)

	# city_map_features = ["UserInfo_2", "UserInfo_4", "UserInfo_7", "UserInfo_19"]
	# digited_city_data = use_map_basis_to_digit(data, features, city_map_basis, city_map_features)

	# save_result(digited_city_data, "digited_city_data.csv", features)

	######################## used to digit province features #####################
	# province_content = load_result("2014中国各省人均可支配收入排行.csv", dir_name = "material_data")
	# province_features = np.array(province_content[0])
	# province_data = np.array(province_content[1:])
	# print(province_features.shape)
	# print(province_data.shape)

	# print(province_features)
	# print(province_data)
	# province_map_basis = create_province_map_basis(province_data, province_features)
	# for k, v in province_map_basis.items():
	# 	print(k, v)


	# contents = load_result("digited_city_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# replace_features = ["UserInfo_6"]
	# data = replace_with_original(data, features, replace_features)

	# print(features.shape)
	# print(data.shape)
	# province_map_features = ["UserInfo_6", "UserInfo_18"]
	# digited_province_data = use_map_basis_to_digit(data, features, province_map_basis, province_map_features)

	# save_result(digited_province_data, "digited_province_data.csv", features)

	#######################  used to sovle the phone #############################
	#### we think that which company is not that importat, whether the phone has is important###
	### so we replace the currrent col with original in 'UserInfo_8'
	# contents = load_result("digited_province_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
	
	# replace_features = ["UserInfo_8"]
	# data = replace_with_original(data, features, replace_features)

	# phone_map_basis = create_phone_map_basis()
	# phone_map_features = ["UserInfo_8"]
	# digited_phone_data = use_map_basis_to_digit(data, features, phone_map_basis, phone_map_features)

	# save_result(digited_phone_data, "digited_phone_data.csv", features)

	##################### used to map marry info ########################
	########### UserInfo_21 ##########
	# contents = load_result("digited_phone_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
	
	# replace_features = ["UserInfo_21"]
	# data = replace_with_original(data, features, replace_features)

	# marrage_map_basis = create_marrage_map_basis()
	# marrage_map_features = ["UserInfo_21"]
	# digited_marrage_data = use_map_basis_to_digit(data, features, marrage_map_basis, marrage_map_features)

	# save_result(digited_marrage_data, "digited_marrage_data.csv", features)


	#################### used to map resident info ##########################
	########### UserInfo_23 #############
	# contents = load_result("digited_marrage_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# resident_feature = ["UserInfo_23"]

	# digited_residence_data = map_residence(data, features, resident_feature)
	# save_result(digited_residence_data, "digited_residence_data.csv", features)



	################### map the str to digit ##################
	contents = load_result("digited_residence_data.csv")
	# remove the feature named 'ListingInfo' 
	features = np.array(contents[0])
	data = np.array(contents[1:])

	delete_fea_pos = ["ListingInfo"]
	data, features, deleted_features = delete_features(data, features, delete_feas_list = delete_fea_pos)

	label_lines = np.array(load_result("train_label_original.csv"))
	from save_load_result import convert_to_float
	label = convert_to_float(label_lines)
	digited_data, features_map_info = map_str_to_digit(data, features, label)

	save_result(digited_data, "after_solve_specialStr_digited_data.csv", features)
	save_result(np.array(features_map_info), "solve_specialStr_features_map_infos.csv", dir_name = "resultData/features_map")
