#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 14:52:54
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

# from get_data import load_data, data_info, print_instance_format, extract_target
# from solve_data import feature_value_class, filling_miss, delete_features, \
# 						sort_features_with_entroy, delete_no_discrimination_features, \
# 						map_str_feature_to_value, convert_to_digit
# import numpy as np
# from save_load_result import save_result, load_result, save_features_info
# from collections import OrderedDict

# if __name__ == '__main__':


# 	train_master_feas, train_master_data, a = load_data("PPD_Training_Master_GBK_3_1_Training_Set.csv")
# 	features, train_master_set, train_label = extract_target(train_master_feas, train_master_data)

# 	save_result(train_master_set, "extractedTarget_originalData.csv", features)

# 	################### how to use filling_miss ###################
# 	new_data = train_master_set.copy()
# 	new_features = features.copy()
# 	delete_fea = []
# 	missing_num = []
# 		#!start from range(1,...) is because the first line of the feature is the id, useless
# 	for fea_pos in range(1, len(features)):
# 		fea_val_cla = feature_value_class(train_master_set, fea_pos, train_label)
# 		if not fea_val_cla["miss"]._present_num == 0:
# 			new_data = filling_miss(new_data, fea_pos, train_label, fea_val_cla, delete_fea, missing_num)
	
# 	new_data, new_features, deleted_feas = delete_features(new_data, features, delete_fea)

# 	save_result(train_label, "train_label_original.csv")
# 	save_result(np.array(missing_num), "deleted_features_with_too_many_missing.csv", np.array(deleted_feas))
# 	save_result(new_data, "data_after_filling_missing.csv", new_features)
	
# 	#################### second step to remove no_discrimination features ######################
# 	contents = load_result("data_after_filling_missing.csv")
# 	features = np.array(contents[0])
# 	data = np.array(contents[1:])

# 	label_lines = np.array(load_result("train_label_original.csv"))

# 	from save_load_result import convert_to_float
# 	label = convert_to_float(label_lines)

# 	index_entroy = sort_features_with_entroy(data, features, label)
# 	new_data, new_features, deleted_features, delete_fea_entroy = delete_no_discrimination_features(data, features, index_entroy)
# 	save_result(np.array(delete_fea_entroy), "deleted_features_with_no_discrimination(entroy).csv", np.array(deleted_features))
# 	save_result(new_data, "data_after_delete_no_discrimination_features.csv", new_features)

# 	save_features_info(new_data, new_features, label, "after_delete_features_infos.csv")

# 	##################### convert str in data into int digit #####################
# 	# contents = load_result("data_after_delete__no_discrimination_features.csv")
# 	# features = np.array(contents[0])
# 	# data = np.array(contents[1:])
# 	# label_lines = np.array(load_result("train_label_original.csv"))
# 	# from save_load_result import convert_to_float
# 	# label = convert_to_float(label_lines)

# 	# features_map_info = list()
# 	# for fea_pos in range(1, len(features)):
# 	# 	map_info = OrderedDict()
# 	# 	feature_map_info = OrderedDict()
# 	# 	fea_val_cla = feature_value_class(data, fea_pos, label)
# 	# 	# if this feature is a string value, just convert it to value
# 	# 	if fea_val_cla["str_feature"]:
# 	# 		data, map_info = map_str_feature_to_value(data, fea_pos, fea_val_cla)
# 	# 		feature_map_info[features[fea_pos]] = map_info
# 	# 		features_map_info.append([feature_map_info])

# 	# digit_data = convert_to_digit(data)

# 	# save_result(digit_data, "after_delete_get_digit_data.csv", features)
# 	# save_result(np.array(features_map_info), "features_map_infos.csv", dir_name = "resultData/features_map")

# 	#################### use PCA to have a first try #########################
# 	