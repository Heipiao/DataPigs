#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-02 22:50:18
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from save_load_result import load_result, convert_to_float
from solve_data import feature_value_class, FeatureInData


# data = np.array([[1,2,3,4,5], [6,7,32432,9,6], [2,3,123,51269,6234]])
# print(data)
# print(data[:, 0])
# fea_mean = np.mean(data[:, 0])
# fea_average = np.var(data[:,0])
# print(fea_mean)
# print(fea_average)

# contents = load_result("after_Str_features_digited_data.csv")
# features = np.array(contents[0])
# data = np.array(contents[1:])
# label_lines = np.array(load_result("train_label_original.csv"))
# print(label_lines.shape)

# label = convert_to_float(label_lines)

# from map_features_to_digit import convert_to_numerical

# data = convert_to_numerical(data, features)


# features_info = feature_value_class(data, 5, label)

# print("num_of_value: ", features_info["num_of_value"])



# label = label.reshape((label.size, ))


# positive_index = np.array([index for index in range(len(label)) if label[index] == 1])
# negitive_index = np.array([index for index in range(len(label)) if label[index] == 0])






# y_positive = data[label == 1, 5]
# y_negitive = data[label == 0, 5]
# plt.xlabel("instances(30000)")
# plt.ylabel("value")
# plt.title(features[5] + " value - label " + "distributed " + "in instances" + \
# 			"\n the arrow marked --> Proportion of positive in that value")


# plt.scatter(positive_index, y_positive, marker = 'o', color = 'r')
# plt.scatter(negitive_index, y_negitive, marker = 'x', color = 'g')


# for k, v in features_info.items():
# 	if isinstance(v, FeatureInData):
# 		arrow_data = round(v._respond_positive_num / v._present_num , 3)
# 		arrow_start_position_x = len(data) + 20
# 		arrow_start_position_y = int(k)
# 		arrow_end_postion_x = arrow_start_position_x
# 		arrow_end_postion_y = int(k) - 0.5
# 		plt.annotate(arrow_data, \
# 					xy=(arrow_start_position_x,arrow_start_position_y), \
# 					xytext=(arrow_end_postion_x,arrow_end_postion_y), \
# 					arrowprops=dict(facecolor='blue', shrink=0.05))

	

# plt.savefig("test.png")

# x = np.array(range(1, 30001))
# y = data[:, 119]
# #print(data[:50, 119])
# print(features[119])

# count_sum_p = 0
# count_sum_n = 0
# count_sum = 0
# draw_x = list()
# draw_y = list()
# x = list()
# y = list()
# for i in range(len(data)):

# 	if data[i, 119] > 1700 and data[i, 119] < 1850:
# 		print(data[i, 119], label[i])
# 		count_sum += 1
# 		if label[i] == 1:
# 			draw_x.append(count_sum + 1)
# 			draw_y.append(data[i, 119])
# 			count_sum_p += 1
# 		else:
# 			x.append(count_sum + 1)
# 			y.append(data[i, 119])
# 			count_sum_n += 1

# print("total", count_sum_p)

# plt.scatter(draw_x, draw_y, marker = 'o', color = 'r', s = 3)
# plt.scatter(x, y, marker = 'x', color = 'k', s = 10)
# plt.savefig("test.png")

# print(x.shape)
# print(y.shape)

# plt.scatter(x, y)

# fea_mean = np.mean(y)

# x1 = np.array(range(-5000, 35000))
# print(x1.shape)
# y_mean = fea_mean * np.ones((x1.size))

# print(y_mean)
# plt.plot(x1, y_mean, color = 'b', linestyle = "--")

# plt.savefig("test.png")


data = np.array([[1,2,3,4], [5,6,7,8], [1,5,6,9], [0, 2,4,6], [4,4,4,9]])
print(data)
print(data[:, 3].shape)

b = data[:, 3]
values_in_fea = np.array(list(set(list(b))))
sorted_fea_data = values_in_fea[np.argsort(values_in_fea)]
print(sorted_fea_data)
print(list(sorted_fea_data))