#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-06 23:10:04
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import numpy as np

from save_load_result import load_result, save_result
from collections import OrderedDict
DATA_DIR = "PPD-First-Round-Data/Training Set/"
SAVE_DIR = "resultData"

#################### load the needed data ####################################
log_info = np.array(load_result("PPD_LogInfo_3_1_Training_Set.csv", dir_name = DATA_DIR))
log_info_features = log_info[0]
log_info_data = log_info[1:]


update_info = np.array(load_result("PPD_Userupdate_Info_3_1_Training_Set.csv", dir_name = DATA_DIR))
update_info_features = update_info[0]
update_info_data = update_info[1:]

id_target = np.array(load_result("PPD_Training_Master_GBK_3_1_Training_Set.csv", dir_name = DATA_DIR))

features = id_target[0, [0, -2, -1]]
data = id_target[1:, [0, -2, -1]]
# # print(features)
# print(data)
# print(log_info_features)
# print(log_info_data[:10])

# print(update_info_features)
# print(update_info_data[:20])

# id_index = np.where(log_info_data[:, 0] == "10001")[0]
# print(id_index)
# for i in id_index:
# 	print(i)

# each id contain:
#	-- id name --> "target" "borrow_date"
#				--> "land_info" --> land_operate_code land_operate_style land_date
#				--> "modify_info" --> modify_info modify_date
#
def combine_land_modify_infos(data, log_info_data, update_info_data):
	all_id_info = OrderedDict()
	for id_pos in range(len(data)):
		id_name = data[id_pos, 0]
		all_id_info[id_name] = OrderedDict()
		all_id_info[id_name]["target"] = data[id_pos, 1]
		all_id_info[id_name]["borrow_date"] = data[id_pos, 2]
		# add the land info
		all_id_info[id_name]["land_info"] = OrderedDict()
		land_date = list()
		land_operate_code = list()
		land_operate_style = list()

		id_index_in_land = np.where(log_info_data[:, 0] == id_name)[0]
		for i in id_index_in_land:

			land_operate_code.append(log_info_data[i, 2])
			land_operate_style.append(log_info_data[i, 3])
			land_date.append(log_info_data[i, 4])
		all_id_info[id_name]["land_info"]["land_operate_code"] = land_operate_code
		all_id_info[id_name]["land_info"]["land_operate_style"] = land_operate_style
		all_id_info[id_name]["land_info"]["land_date"] = land_date

		# add the modify info
		all_id_info[id_name]["modify_info"] = OrderedDict()
		modify_info = list()
		modify_date  =list()
		id_index_in_modify = np.where(update_info_data[:, 0] == id_name)[0]
		for i in id_index_in_modify:
			modify_info.append(update_info_data[i, 2])
			modify_date.append(update_info_data[i, 3])

		all_id_info[id_name]["modify_info"]["modify_things"] = modify_info
		all_id_info[id_name]["modify_info"]["modify_date"] = modify_date
	save_result(all_id_info, "all_id_info.pickle", dir_name = SAVE_DIR)


if __name__ == '__main__':

	#combine_land_modify_infos(data, log_info_data, update_info_data)
	combined = load_result("all_id_info.pickle", dir_name = SAVE_DIR)
	print(combined["10001"])
