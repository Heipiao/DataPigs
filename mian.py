#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 14:52:54
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

from get_data import load_data, data_info, print_instance_format, extract_target
from solve_data import feature_value_class, filling_miss, delete_features
import numpy as np

if __name__ == '__main__':
#	train_master_feas, train_master_data, a = load_data("PPD_Training_Master_GBK_3_1_Training_Set.csv")
	#train_master_feas, train_master_data, a = load_data("PPD_LogInfo_3_1_Training_Set.csv")
	# print("***********just view features and one data: ***********")
	# print(train_master_feas)

	# print(train_master_data[0])

	# print("*****print correspondence between features and data")
	# print_instance_format(train_master_feas, train_master_data[0])

	# print("\n***** the info of the features *******")
	# info = data_info(a)
	# for k, v in info.items():
	# 	print(k, v)

	train_master_feas, train_master_data, a = load_data("PPD_Training_Master_GBK_3_1_Training_Set.csv")
	features, train_master_set, train_label = extract_target(train_master_feas, train_master_data)
	print(features)
	print(train_master_set[0:2])
	'''
	print(train_master_set[0:2])
	new_data = train_master_set.copy()
	new_features = features.copy()
	delete_fea = []
	for fea_pos in range(len(new_features)):
		fea_val_cla = feature_value_class(train_master_set, fea_pos, train_label)
		new_data, new_features = filling_miss(new_data, new_features, fea_pos, train_label, fea_val_cla, delete_fea)
	print(delete_fea)

	print(new_data[0:2])
	'''
	new_data = train_master_set.copy()
	new_features = features.copy()
	delete_fea = []
	for fea_pos in range(1, len(features)):
		fea_val_cla = feature_value_class(train_master_set, fea_pos, train_label)
		if not fea_val_cla["miss"]._present_num == 0:
			new_data = filling_miss(new_data, fea_pos, train_label, fea_val_cla, delete_fea)
	
	new_data, new_features, deleted_feas = delete_features(train_master_set, features, delete_fea)
	print(new_features)
	print(new_data[0:2])
	print(deleted_feas)