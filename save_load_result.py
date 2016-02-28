#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-28 15:41:31
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: save my result in this area
#	SAVE_LOAD_AREA = ./resultData/
'''
with open('eggs.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
    spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
'''

import os
import csv
import collections
import numpy as np
# all the input should be array 
# save the data as csv in SAVE_LOAD_AREA/file_name
def save_result(data, file_name, features = 10, dir_name = "resultData"):

	file_path = os.path.join(os.getcwd(), dir_name, file_name)
	with open(file_path, "w", newline='') as csv_file:
		spamwriter = csv.writer(csv_file)
		if isinstance(features, collections.Iterable):
			spamwriter.writerow(features)
		if data.ndim == 1:
			spamwriter.writerow(data)
		else:	
			for i in range(len(data)):
				spamwriter.writerow(data[i])

# save the result of function called 'feature_value_class' definited in solve_data.py
#	into the file
# the order for each key is according to the comment above the 'feature_value_class'
# str_feature
# 2, 3 --> different according to str_feature`s value
# num_positive
# num_negitive
# class FeatureInData`s instances
#	- _present_num
#	- _respond_positive_num
#	- _respond_negitive_num
def save_features_info(data, features, label, file_name, dir_name = "resultData"):

	file_path = os.path.join(os.getcwd(), dir_name, file_name)
	first_line = np.array(['features_name', 'str_feature', \
						'postitive(average|most_present)', \
						'negitive(average|most_present)', \
						'num_positive', \
						'num_negitive', 'feature_value_info'])

	with open(file_path, "w", newline='') as csv_file:
		spamwriter = csv.writer(csv_file)
		spamwriter.writerow(first_line)

	from solve_data import feature_value_class, FeatureInData
	from collections import OrderedDict
	for fea_pos in range(1, len(features)):
		feature_info = list()
		feature_info.append(features[fea_pos])	
		fea_val_cla = feature_value_class(data, fea_pos, label)
		feature_info.append(fea_val_cla["str_feature"])
		if fea_val_cla["str_feature"]:
			feature_info.append(fea_val_cla["most_presentS_positive"])
			feature_info.append(fea_val_cla["most_presentS_negitive"])
		else:
			feature_info.append(fea_val_cla["average_positive"])
			feature_info.append(fea_val_cla["average_negitive"])
		feature_info.append(fea_val_cla["num_positive"])
		feature_info.append(fea_val_cla["num_negitive"])
		for k, v in fea_val_cla.items():
			if isinstance(v, FeatureInData):
				value_info = OrderedDict()
				value_info["value"] = k
				value_info["present_num"] = v._present_num
				value_info["respond_positive_num"] = v._respond_positive_num
				value_info["respond_negitive_num"] = v._respond_negitive_num
				feature_info.append(value_info)
		with open(file_path, "a+", newline='') as csv_file:
			spamwriter = csv.writer(csv_file)
			spamwriter.writerow(feature_info)



# load the data and features
def load_result(file_name, dir_name = "resultData"):
	data_file_path = os.path.join(os.getcwd(), dir_name, file_name)
	if os.path.exists(data_file_path):
		with open(data_file_path, newline='') as csv_file:
			csv_reader = csv.reader(csv_file)
			lines = [line for line in csv_reader]
	return lines

# note: if you want to get the label data as a float type please call this function
#	-- the input: should be array types with size (row, 1)
def convert_to_float(strf_array):
	convert_result = np.array(list(map(float, strf_array[:, 0])))
	convert_result = convert_result.reshape(len(convert_result), 1)
	return convert_result