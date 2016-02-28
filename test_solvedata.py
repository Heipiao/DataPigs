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


data = np.array([[1,2,3,4,5],
				[1,2,4,5,3],
				[2,2,4,5,3],
				[3,2,4,5,1],
				[1,2,3,4,5]])
features = np.array([1, 2, 3, 4, 5])
label = np.array([[1],
				[1],
				[0],
				[1],
				[0]])

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
print(result_entroy)

for k, v in result_entroy.items():
	print(k, v)

print(round(0.05, 1))