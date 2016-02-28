#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 11:03:01
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import numpy as np

# data = np.array([[1,2,3,4,5], [6,7,8,9,0], [2,3,4,5,6], [3,3,3,3,3]])
# fea = np.array([1, 2, 3, 4, 5])
# print(data)
# print(fea.shape)

# print(len(fea))

# print(fea == 4)

# print(data[ : ,fea == 4])

# for i in range(len(data)):
# 	print(data[i][1])

# from solve_data import feature_value_class

# from get_data import load_data, extract_target
# train_master_feas, train_master_data, a = load_data("PPD_Training_Master_GBK_3_1_Training_Set.csv")
# features, train_master_set, train_label = extract_target(train_master_feas, train_master_data)
# print(train_label.shape)
# feature_value_class(train_master_set[0:100], 2, train_label)


# from collections import defaultdict
# class test(object):
# 	def __init__(self):
# 		self._prr = "0"
# 	def present(self):
# 		self._prr = "1"
# 	def show(self):
# 		print("sadfds: ", self._prr)


# class TestDict(dict):
# 	def __missing__(self, key):
# 		return test()

# a = defaultdict(test)

# a["dsf"].present()

# a["dsf"].show()

data = np.array([['E','F','C',4,5], [6,7,'D','9',6], [2,3,'C',5,6], [3,8,'D',3,3], [2,2,'',5,6]])
fea = np.array([1, 2, '3', 4, 5])
label = np.array([['1'], ['0'], ['0'], ['1'], ['1']])

print(label.shape)
label = np.array(list(map(int, label[:, 0]))).reshape(len(label), 1)
print(label.shape)

print(fea)
print(data)
print(label)


'''
from solve_data import delete_features
delete_pos = [2, 4]

new_data, new_features, deleted_f = delete_features(data, fea, delete_pos)
print("new data \n")
print(new_features)
print(new_data)
print(deleted_f)
'''

from solve_data import feature_value_class, FeatureInData, filling_miss
from save_load_result import save_result

fea_value_sat = feature_value_class(data, 2, label)
for k,v in fea_value_sat.items():
	if isinstance(v, FeatureInData):
		print(k)
		v.show()
	else:
		print("\n total: ")
		print(k, v)

delete_fea = []
data = filling_miss(data, 2, label, fea_value_sat, delete_fea)

print(data)
print(delete_fea)

save_result(data, fea, "test.csv")