#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-28 18:17:56
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 
import numpy as np
# from functools import reduce

# from solve_data import calculate_feature_entroy, feature_value_class, FeatureInData, \
# 						sort_features_with_entroy, delete_features, calculate_feature_entroy_new
# from save_load_result import save_features_info, write_to_deleted_features_area, \
# 							load_result

# data = np.array([[1,3,2,4,5],
# 				[1,4,2,5,3],
# 				[2,4,2,5,3],
# 				[3,4,2,5,1],
# 				[1,3,2,4,5]])
# features = np.array([1, 2, 3, 4, 5])
# label = " "


# fea_info = feature_value_class(data, 3, label)
# print(calculate_feature_entroy_new(fea_info, len(data)))


# from math import log
# log2 = lambda x : log(x) / log(2)
# v1 = log2(5/2) * (2/5)
# print(v1)
# v2 = log2(5/3) * (3/5)
# print(v2)

# print(v1 + v2)

# print(log2(2))


import matplotlib as mpl
import matplotlib.pyplot as plt
dic = {1:3.4, 6:5.7, 5:2.5, 2:4.4, 7:1.2, 3:0, 4:1.5}
x = np.array(list(dic.keys()))
y = np.array(list(dic.values()))
print(x)
print(y)

plt.scatter(x, y)
plt.xlabel("index of features(1-->total)")
plt.ylabel("measure for each")

plt.savefig("test_show.png")
plt.close()
# features = list(features)
# print(features)
# print(features.shape)
# label = np.array([[0],
# 				[0],
# 				[1],
# 				[0],
# 				[1]])

# adf = list([1, 2])
# asdf = list([4, 5])
# data, features = delete_features(data, features, adf, asdf)
# fea_value_sat = feature_value_class(data, 3, label)
# for k, v in fea_value_sat.items():
# 	print("K = ", k)
# 	if isinstance(v, FeatureInData):
# 		v.show()
# 	else:
# 		print(v)
#save_features_info(data, features, label, "test_save_featuers_info.csv")

# features = np.array(["a", "b", "c", "d", "e"])
# write_to_deleted_features_area(features)#, re_write = True)

# ff = load_result("all_deleted_features.csv")
# def iter_extend(x, y):
# 	return x.extend(y)
# print(ff)
# print(type(ff))
# a = np.array(ff)

# print(a.reshape((a.size, )))
#print(np.array(ff))
# print(features)
# kn = np.array(["c", "d"])
# from solve_data import get_known_features_index
# idex = get_known_features_index(features, kn)
# print(idex)
# result = calculate_feature_entroy(fea_value_sat)
# print(result)
# result_entroy = sort_features_with_entroy(data, features, label)
# for k, v in result_entroy.items():
# 	print(k, v)

# ll = ""
# if ll:
# 	print("dsffffffff")
# ll = sorted(result_entroy.items(), key=lambda d: d[1])
# print(ll)
# print(len(ll))
# print(type(ll))
# print(type(ll[0][0]))
# print(ll[0][0])
# print(round(0.05, 1))

