#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-29 20:30:04
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import numpy as np
from map_features_to_digit import map_str_to_digit_with_experience
from save_load_result import load_result
# data = np.array([['1','2.0','3.0','4','5'], ['6','7','32432','9','6'], ['2','3','123','51269','6234']])
# new_col = np.array([[3, 5, 0]])
# print(data)
# print(new_col)

# print(data[:, 2].shape)

# data[:, 2] = new_col

# print(data)

# import re
# # 0
# pattern_bad = re.compile(r".*市.*|.*自治区.*")
# # 1
# pattern_normal = re.compile(r".*州.*|.*县.*|.*区.*|.*旗.*")
# # 2
# pattern_good1 = re.compile(r".*镇.*|.*乡.*|.*市.*.*庄.*|.*村.*")
# pattern_good2 = re.compile(r".*路.*|.*街.*|.*巷.*|.*村.*")
# # 3
# pattern_great = re.compile(r".*\d.*")

# my_str = "新疆维吾尔自治区喀什地区英吉沙县"
# if pattern_bad.search(my_str):
# 	print(my_str)
if __name__ == '__main__':
	
	# contents = load_result("after_delete_strong_correlation_features_data.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# from map_features_to_digit import convert_to_numerical

	# data = convert_to_numerical(data, features)


	# from map_features_to_digit import map_str_to_digit_with_experience

	# map_str_to_digit_with_experience(data, features, None)
	data = np.array([['1','2.0','3.0','4','5'], ['6','7','32432','9','6'], ['2','3','123','51269','6234']])
	print(data)
	f = [1, 2]
	data1 = np.delete(data, f, axis = 0)
	print(data1)