#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-05 08:40:14
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 


from main_for_process_data import load_data_for_solve, replace_miss, strStyle_features_to_digit
from save_load_result import save_features_info, save_result, load_all_deleted_features_during_train
from solve_data import delete_features
import numpy as np



if __name__ == '__main__':
	data, features, label = load_data_for_solve("PPD_Master_GBK_2_Test_Set.csv", for_train = False)

	data, features = replace_miss(data, features, label, for_train = False)
	#save_result(data, "test/data_after_filling_missing_.csv", features)

	deleted_features_in_train = load_all_deleted_features_during_train(deleted_features_file_label = "deleted_features_with_too_many_missing")
	data, features, deleted = delete_features(data, features, delete_feas_list = deleted_features_in_train)
	save_result(data, "test_data_after_deleted_features.csv", features, dir_name = "resultData/test/")

	data = strStyle_features_to_digit(data, features, for_train = False, use_experience = True)
	save_result(data, "data_after_digited.csv", features, dir_name= "resultData/test/")
	save_features_info(data, features, label, "info_after_digit_all_features.csv", \
						dir_name = "resultData/test/")


