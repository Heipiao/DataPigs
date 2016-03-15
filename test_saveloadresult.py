#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-28 15:49:28
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import numpy as np
import csv
import os
from save_load_result import save_result, load_result, load_all_deleted_features_during_train


# data = np.array([['E','F','C',4,5], [6,7,'D','9',6], [2,3,'C',5,6], [3,8,'D',3,3], [2,2,'',5,6]])
# fea = np.array([1, 2, '3', 4, 5])
# label = np.array([['1'], ['0'], ['0'], ['1'], ['1']])

# save_result(data, "firstTry.csv")

# my_data = ['asdf', 'sdf', 'asdf', '1', '6']
# ll = dict()
# ll["ok"] = 5
# ll["okokok"] = 6
# my_data.append(ll)

# contents = load_result("data_after_delete_no_discrimination_features.csv")
# features = np.array(contents[0])
# print(features)


# fixed_str_features = np.array(load_result("str_features.csv")[0])
# print(fixed_str_features.shape)
# print(fixed_str_features)
# indexs = list()

# for i in range(len(fixed_str_features)):
# 	try:
# 		finded = np.where(features == fixed_str_features[i])[0][0]
# 		indexs.append(finded)
# 	except:
# 		pass

# print(indexs)
all_deleted_features = np.array(load_all_deleted_features_during_train())
print(all_deleted_features.shape)
