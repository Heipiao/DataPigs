#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-01 16:18:22
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

from save_load_result import load_result
from solve_data import feature_value_class, convert_to_digit
import numpy as np

contents = load_result("after_solve_specialStr_digited_data.csv")
features = np.array(contents[0])
data = np.array(contents[1:])
data = convert_to_digit(data)

label_lines = np.array(load_result("train_label_original.csv"))

from save_load_result import convert_to_float, save_features_info
label = convert_to_float(label_lines)

#print(features)

age_feature = "UserInfo_17"
fea_pos = np.where(features == age_feature)[0][0]

age_feature_info = feature_value_class(data, fea_pos, label)

t1 = age_feature_info["average_positive"] * age_feature_info["num_positive"]
t2 = age_feature_info["average_negitive"] * age_feature_info["num_negitive"]
average = (t1 + t2) / (age_feature_info["num_positive"] + age_feature_info["num_negitive"])

print(average)
my_sum = 0
for i in range(len(data)):
	my_sum += data[i, fea_pos]
print("average = ", my_sum / len(data))

print(np.where(data[:, fea_pos] == 0))
print(data[2691, fea_pos])
print(max(data[:, fea_pos]))
print(min(data[:, fea_pos]))