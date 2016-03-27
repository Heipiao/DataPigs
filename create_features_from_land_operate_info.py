#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-25 00:59:24
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import numpy as np

from save_load_result import load_result, save_result
from collections import OrderedDict

from combine_land_update_info import combine_land_modify_infos
import pickle
import pprint

import datetime
	

def get_respond_data(for_train = True):
	if for_train:
		log_file_name = "PPD_LogInfo_3_1_Training_Set.csv"
		update_file_name = "PPD_Userupdate_Info_3_1_Training_Set.csv"
		id_target_file_name = "PPD_Training_Master_GBK_3_1_Training_Set.csv"
		DATA_DIR = "PPD-First-Round-Data/Training Set/"
	else:
		log_file_name = "PPD_LogInfo_2_Test_Set.csv"
		update_file_name = "PPD_Userupdate_Info_2_Test_Set.csv"
		id_target_file_name = "PPD_Master_GBK_2_Test_Set.csv"
		DATA_DIR = "PPD-First-Round-Data/Test Set/"

	#################### load the needed data ####################################
	log_info = np.array(load_result(log_file_name, dir_name = DATA_DIR))
	log_info_features = log_info[0]
	log_info_data = log_info[1:]


	update_info = np.array(load_result(update_file_name, dir_name = DATA_DIR))
	update_info_features = update_info[0]
	update_info_data = update_info[1:]

	id_target = np.array(load_result(id_target_file_name, dir_name = DATA_DIR))

	if for_train:
		features = id_target[0, [0, -2, -1]]
		data = id_target[1:, [0, -2, -1]]
	else:
		features = id_target[0, [0, -1]]
		data = id_target[1:, [0, -1]]

	return data, log_info_data, update_info_data


def make_operability(user_log_info):
	borrowed_date = list(map(int, user_log_info["borrow_success_date"].split("/")))
	operate_code = list(map(int, user_log_info["land_info"]["land_operate_code"]))
	operate_style = list(map(int, user_log_info["land_info"]["land_operate_style"]))
	all_land_date = [list(map(int, date.split("-"))) for date in user_log_info["land_info"]["land_date"]]

	return borrowed_date, operate_code, operate_style, all_land_date

def find_most_appear(input_list):
	if input_list:

		appear_info = list(map(lambda x: input_list.count(x), input_list))
		most_appear_num = max(appear_info)
		most_appear_indexs = [pos for pos, value in enumerate(appear_info) if value == most_appear_num]

		most_appear = input_list[most_appear_indexs[0]]
		# print(appear_info)
		# print(most_appear_indexs)
		# print(most_appear)
	else:
		most_appear = 0
		most_appear_num = 0
		most_appear_indexs = []
	return most_appear, most_appear_num, most_appear_indexs

def find_value_index(input_list, value):
	value_indexs = [p for p, v in enumerate(input_list) if v == value]
	return value_indexs

# input : user_log_info = combine_data[user]  user_id = key of combine_data
# output: show below in the ### create features from land info ### part
def create_features_from_whole_log_info(user_log_info):

	borrowed_date, operate_code, operate_style, all_land_date = make_operability(user_log_info)

	count_land_frequent = len(all_land_date)
	count_land_operate_code = len(set(operate_code))
	count_land_operate_style = len(set(operate_style))
	most_appeared_operate_code, count_most_appeared_operate_code, indexs = find_most_appear(operate_code)
	most_appeared_operate_style, count_most_appeared_operate_style, indexs = find_most_appear(operate_style)

	user_whole_log_info = OrderedDict()
	user_whole_log_info["count_land_frequent"] = count_land_frequent
	user_whole_log_info["count_land_operate_code"] = count_land_operate_code 
	user_whole_log_info["count_land_operate_style"] = count_land_operate_style 
	user_whole_log_info["most_appeared_operate_code"] = most_appeared_operate_code 
	user_whole_log_info["most_appeared_operate_style"] = most_appeared_operate_style 
	user_whole_log_info["count_most_appeared_operate_code"] = count_most_appeared_operate_code
	user_whole_log_info["count_most_appeared_operate_style"] = count_most_appeared_operate_style

#	print(user_whole_log_info)

	return user_whole_log_info

#	- count_land_frequent_near_borrowed: count land frequent 15 days near borrowed
#	- count_operate_code_near_borrowed
#	- count_operate_style_near_borrowed
#   - most_appeared_operate_code_near_borrowed
#	- most_appeared_operate_style_near_borrowed
#	- count_most_appeared_operate_code_near_borrowed
#	- count_most_appeared_operate_style_near_borrowed

def find_date_near_borrowed_day(borrowed_date, all_land_date):
	d_b = datetime.datetime(borrowed_date[0],borrowed_date[1],borrowed_date[2])
	near_date_indexs = [p for p, date in enumerate(all_land_date) \
					if (d_b - datetime.datetime(date[0],date[1],date[2])).days < 30 and 
						not (d_b - datetime.datetime(date[0],date[1],date[2])).days == 0]
	#sorted_all_land_date = sorted(all_land_date)

	return near_date_indexs
# 'neared' means that the log date is 7 days within the borrowed_date
def create_features_from_near_log_info(user_log_info):
	borrowed_date, operate_code, operate_style, all_land_date = make_operability(user_log_info)
	near_log_date_indexs = find_date_near_borrowed_day(borrowed_date, all_land_date)

	near_day_operate_code = [operate_code[i] for i in near_log_date_indexs]
	near_day_operate_style = [operate_style[i] for i in near_log_date_indexs]

	count_land_frequent_near_borrowed = len(near_log_date_indexs)
	count_operate_code_near_borrowed = len(set(near_day_operate_code))
	count_operate_style_near_borrowed = len(set(near_day_operate_style))
	most_appeared_operate_code_near_borrowed, count_most_appeared_operate_code_near_borrowed, indexs = find_most_appear(near_day_operate_code)
	most_appeared_operate_style_near_borrowed, count_most_appeared_operate_style_near_borrowed, indexs = find_most_appear(near_day_operate_style)

	user_near_day_log_info = OrderedDict()
	user_near_day_log_info["count_land_frequent_near_borrowed"] = count_land_frequent_near_borrowed
	user_near_day_log_info["count_operate_code_near_borrowed"] = count_operate_code_near_borrowed 
	user_near_day_log_info["count_operate_style_near_borrowed"] = count_operate_style_near_borrowed 
	user_near_day_log_info["most_appeared_operate_code_near_borrowed"] = most_appeared_operate_code_near_borrowed 
	user_near_day_log_info["most_appeared_operate_style_near_borrowed"] = most_appeared_operate_style_near_borrowed 
	user_near_day_log_info["count_most_appeared_operate_code_near_borrowed"] = count_most_appeared_operate_code_near_borrowed
	user_near_day_log_info["count_most_appeared_operate_style_near_borrowed"] = count_most_appeared_operate_style_near_borrowed

#	print(user_near_day_log_info)

	return user_near_day_log_info

def create_features_from_borrowed_log_info(user_log_info):
	borrowed_date, operate_code, operate_style, all_land_date = make_operability(user_log_info)

	borrowed_date_indexs = find_value_index(all_land_date, borrowed_date)

	borrowed_day_operate_code = [operate_code[i] for i in borrowed_date_indexs]
	borrowed_day_operate_style = [operate_style[i] for i in borrowed_date_indexs]

	count_land_frequent_borrowed_day = len(borrowed_date_indexs)
	count_operate_code_borrowed_day = len(set(borrowed_day_operate_code))
	count_operate_style_borrowed_day = len(set(borrowed_day_operate_style))
	most_appeared_operate_code_borrowed_day, count_most_appeared_operate_code_borrowed_day, indexs = find_most_appear(borrowed_day_operate_code)
	most_appeared_operate_style_borrowed_day, count_most_appeared_operate_style_borrowed_day, indexs = find_most_appear(borrowed_day_operate_style)

	user_borrowed_day_log_info = OrderedDict()
	user_borrowed_day_log_info["count_land_frequent_borrowed_day"] = count_land_frequent_borrowed_day
	user_borrowed_day_log_info["count_operate_code_borrowed_day"] = count_operate_code_borrowed_day 
	user_borrowed_day_log_info["count_operate_style_borrowed_day"] = count_operate_style_borrowed_day 
	user_borrowed_day_log_info["most_appeared_operate_code_borrowed_day"] = most_appeared_operate_code_borrowed_day 
	user_borrowed_day_log_info["most_appeared_operate_style_borrowed_day"] = most_appeared_operate_style_borrowed_day 
	user_borrowed_day_log_info["count_most_appeared_operate_code_borrowed_day"] = count_most_appeared_operate_code_borrowed_day
	user_borrowed_day_log_info["count_most_appeared_operate_style_borrowed_day"] = count_most_appeared_operate_style_borrowed_day

#	print(user_borrowed_day_log_info)

	return user_borrowed_day_log_info


def land_date_min_max(all_land_date):
	if all_land_date:
		# land_date_quantization
		sorted_date = sorted(all_land_date)
		first_landed = sorted_date[0]
		lastest_landed = sorted_date[-1]

		date1 = datetime.datetime(first_landed[0],first_landed[1],first_landed[2])
		date2 = datetime.datetime(lastest_landed[0],lastest_landed[1],lastest_landed[2])
		land_span = (date2 - date1).days
	else:
		first_landed = 0
		lastest_landed = 0
		land_span = 0
	return first_landed, lastest_landed, land_span


def create_features_from_first_log_info(user_log_info):
	borrowed_date, operate_code, operate_style, all_land_date = make_operability(user_log_info)
	first_landed, lastest_landed, land_span = land_date_min_max(all_land_date)
	calculate_land_span = land_span

	first_log_date_indexs = find_value_index(all_land_date, first_landed)

	first_log_operate_code = [operate_code[i] for i in first_log_date_indexs]
	first_log_operate_style = [operate_style[i] for i in first_log_date_indexs]

	count_land_frequent_first_log = len(first_log_date_indexs)
	count_first_land_operate_code = len(set(first_log_operate_code))
	count_first_land_operate_style = len(set(first_log_operate_style))
	most_apperaed_opearte_code_first_land, count_most_appeared_operate_code_first_land, indexs = find_most_appear(first_log_operate_code)
	most_apperaed_opearte_style_first_land, count_most_appeared_operate_style_first_land, indexs = find_most_appear(first_log_operate_style)

	user_first_log_info = OrderedDict()
	user_first_log_info["calculate_land_span"] = calculate_land_span
	user_first_log_info["count_land_frequent_first_log"] = count_land_frequent_first_log 
	user_first_log_info["count_first_land_operate_code"] = count_first_land_operate_code 
	user_first_log_info["count_first_land_operate_style"] = count_first_land_operate_style 
	user_first_log_info["most_apperaed_opearte_code_first_land"] = most_apperaed_opearte_code_first_land 
	user_first_log_info["most_apperaed_opearte_style_first_land"] = most_apperaed_opearte_style_first_land
	user_first_log_info["count_most_appeared_operate_code_first_land"] = count_most_appeared_operate_code_first_land
	user_first_log_info["count_most_appeared_operate_style_first_land"] = count_most_appeared_operate_style_first_land
#	print(user_first_log_info)

	return user_first_log_info
################################## create features from land info ######################
# the structure is:
#	Users is a OrderDict() that contains all the users, its key is user`s id, 
#		its key correspond value is a OrderDict() that contains extracted features.
#		extract faetures name as a key, while result as correspond key`s value
# 	Users[user_id] = {"extracted_faetures":features_info, ....}
# the extracted features is:

#(create_features_from_whole_log_info)
#	- count_land_frequent:	the number of whole landed frequent
#	- count_land_operate_code : the number of different value of operate code appear when landing 
#	- count_land_operate_style : the number of different value of operate style appear when landing 
#	- most_appeared_operate_code
#	- most_appeared_operate_style
#	- count_most_appeared_operate_code
#	- count_most_appeared_operate_style

#(create_features_from_neared_log_info)
#	- count_land_frequent_near_borrowed: count land frequent 15 days near borrowed
#	- count_operate_code_near_borrowed
#	- count_operate_style_near_borrowed
#   - most_appeared_operate_code_near_borrowed
#	- most_appeared_operate_style_near_borrowed
#	- count_most_appeared_operate_code_near_borrowed
#	- count_most_appeared_operate_style_near_borrowed

#(create_features_from_borrowed_log_info)
#	- count_land_frequent_borrowed_day: meaning is just showed as the name
#	- count_operate_code_borrowed_day
#	- count_operate_style_borrowed_day
#	- most_appeared_operate_code_borrowed_day
#	- most_appeared_operate_style_borrowed_day
#	- count_most_appeared_operate_code_borrowed_day
#	- count_most_appeared_operate_style_borrowed_day
#	
#(create_features_from_first_log_info)	
#	- calculate_land_span: time between first land and lastest land
#	- count_land_frequent_first_log
#	- count_first_land_operate_code: the number of different operate code value 
#	- count_first_land_operate_style: ..
#	- most_apperaed_opearte_code_first_land
#	- most_apperaed_opearte_style_first_land
#	- count_most_appeared_operate_code_first_land
#	- count_most_appeared_operate_style_first_land

def create_features_from_log_info(combine_data):
	users_land_features = OrderedDict()
	for user in combine_data.keys():
		if not int(user) in users_land_features.keys():
			users_land_features[int(user)] = OrderedDict()
		whole_log_features = create_features_from_whole_log_info(combine_data[user])
		borrowed_log_features = create_features_from_borrowed_log_info(combine_data[user])
		first_log_features = create_features_from_first_log_info(combine_data[user])
		near_log_features = create_features_from_near_log_info(combine_data[user])

		dictMerged1=OrderedDict(whole_log_features, **borrowed_log_features)
		dictMerged1=OrderedDict(dictMerged1, **first_log_features)
		users_land_features[int(user)] = OrderedDict(dictMerged1, **near_log_features)

	return users_land_features

# 5357
if __name__ == '__main__':
	# data, log_info_data, update_info_data = get_respond_data(for_train = False)
	# combine_land_modify_infos(data, log_info_data, update_info_data, saved_dir = "resultData/test/")

	saved_dir = "resultData"
	combined_info_file = os.path.join(saved_dir, "all_id_info.pickle")
	fid=open(combined_info_file, "rb")

	data=OrderedDict(pickle.load(fid))
	fid.close()

	pprint.pprint(data["10001"])
	whole_log_features = create_features_from_whole_log_info(data["10001"])
	borrowed_log_features = create_features_from_borrowed_log_info(data["10001"])
	first_log_features = create_features_from_first_log_info(data["10001"])
	near_log_features = create_features_from_near_log_info(data["10001"])

	dictMerged1=OrderedDict(whole_log_features, **borrowed_log_features)
	dictMerged1=OrderedDict(dictMerged1, **first_log_features)
	dictMerged1=OrderedDict(dictMerged1, **near_log_features)
	print(dictMerged1)
	print(list(dictMerged1))