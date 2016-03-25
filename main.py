#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-18 20:53:29
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

from main_for_process_data import load_data_for_solve, replace_miss

import numpy as np
from solve_data import delete_features
from save_load_result import save_result, load_result, load_all_deleted_features_during_train
from map_features_to_digit import convert_to_numerical

from create_new_features import solve_user_info_package
from create_features_from_weblog import solve_weblog_info_package
from create_features_from_thirdpart import solve_thirdparty_info_package
from create_features_log_update import extract_log_update_package



##### what you should do is only:
### 
# use the parameter:
#		for_train: to control train or test which will load different data set
#		saved_area: to control where to solve your result

def pipeline_for_features_solved(for_train = True, saved_area = "resultData"):
	if for_train:
		print("**************** Train ************************")
		data_file_name = "PPD_Training_Master_GBK_3_1_Training_Set.csv"
	else:
		print("**************** Test ************************")
		data_file_name = "PPD_Master_GBK_2_Test_Set.csv"
	data, features, label = load_data_for_solve(data_file_name, for_train)
	data, features = replace_miss(data, features, label, for_train)

	deleted_features_in_train = load_all_deleted_features_during_train(deleted_features_file_label = "deleted_")
	data, features, deleted = delete_features(data, features, delete_feas_list = deleted_features_in_train)
	print(deleted)

	data, features = solve_user_info_package(data, features, saved_dir = saved_area)

	data, features = solve_weblog_info_package(data, features, saved_dir = saved_area)

	data, features = solve_thirdparty_info_package(data, features, saved_dir = saved_area)


	data, features = extract_log_update_package(data, features, for_train)

	return data, features


if __name__ == '__main__':
	# "resultData/test"
	data, features = pipeline_for_features_solved(for_train = False, saved_area = "resultData/test/")
	print(data.shape)
	save_result(data, "data_after_features_processed.csv", features, dir_name = "resultData/test")
