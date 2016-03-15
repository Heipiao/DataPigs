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
	save_result(data, "test/after_filling_missing_data.csv", features)

	deleted_features_in_train = load_all_deleted_features_during_train()
	data, features, deleted = delete_features(data, features, delete_feas_list = deleted_features_in_train)


	data = strStyle_features_to_digit(data, features, for_train = False, use_experience = True)
	save_features_info(data, features, label, "test/after_digit_all_features_info.csv")


