#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 14:52:13
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from solve_data import feature_value_class
from save_load_result import load_result, get_known_features_index
# view each feature`s value distributed
def view_each_features(data, features):
	str_style_features = np.array(load_result("str_features.csv")[0])
	no_draw_features_index = get_known_features_index(features, str_style_features)



	for fea_pos in range(1, len(features) - 1):

		# not draw the str style features
		if not fea_pos in no_draw_features_index:
			# print(fea_pos)
			# print(features[fea_pos])
			x = np.array(range(1, 30001))
			y = data[:, fea_pos]
			plt.scatter(x, y)
			

			feature_name = features[fea_pos]
			plt.xlabel("instances(30000)")
			plt.ylabel("value")
			plt.title(feature_name + " value " + "distributed " + "in instances")

			# rect = plt.bar(left = (0,1),height = (0.5,0.5),width = 0.15)
			# plt.legend((rect,),(feature_name + "`s value",))

			file_path = "view_data_area/" + "(" + str(fea_pos) + ")" + feature_name +  ".png"
			plt.savefig(file_path)
			plt.close()







if __name__ == '__main__':
	contents = load_result("after_Str_features_digited_data.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	from map_features_to_digit import convert_to_numerical

	data = convert_to_numerical(data, features)
	view_each_features(data, features)