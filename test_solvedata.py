#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-28 18:17:56
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 
import numpy as np


from solve_data import calculate_feature_entroy, feature_value_class, FeatureInData, sort_features_with_entroy


data = np.array([[1,3,2,4,5],
				[1,4,2,5,3],
				[2,4,2,5,3],
				[3,4,2,5,1],
				[1,3,2,4,5]])
features = np.array([4, 1, 4, 3, 1])
label = np.array([[0],
				[0],
				[1],
				[0],
				[1]])

# fea_value_sat = feature_value_class(data, 2, label)
# for k, v in fea_value_sat.items():
# 	print("K = ", k)
# 	if isinstance(v, FeatureInData):
# 		v.show()
# 	else:
# 		print(v)

# result = calculate_feature_entroy(fea_value_sat)
# print(result)
result_entroy = sort_features_with_entroy(data, features, label)
for k, v in result_entroy.items():
	print(k, v)

ll = ""
if ll:
	print("dsffffffff")
# ll = sorted(result_entroy.items(), key=lambda d: d[1])
# print(ll)
# print(len(ll))
# print(type(ll))
# print(type(ll[0][0]))
# print(ll[0][0])
# print(round(0.05, 1))

