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

from solve_data import get_known_features_index
import os
import re
import csv
import collections
import numpy as np
import pickle
# all the input should be array 
# save the data as csv in SAVE_LOAD_AREA/file_name
def save_result(data, file_name, features = 10, style = "w",dir_name = "resultData"):

	file_path = os.path.join(os.getcwd(), dir_name, file_name)
	pickle_pattern = re.compile(r".*\.pickle$")
	# in order to save like this, you should named the file like: ****.pickle
	if pickle_pattern.search(file_name):
		with open(file_path, "wb") as pickle_f:
			pickle.dump(data, pickle_f)
	else:
		with open(file_path, style, newline='') as csv_file:
			spamwriter = csv.writer(csv_file)
			if isinstance(features, collections.Iterable):
				spamwriter.writerow(features)
			try:
				if data.ndim == 1:
					spamwriter.writerow(data)
				else:	
					for i in range(len(data)):
						spamwriter.writerow(data[i])
			except:
				spamwriter.writerow(data)

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
						'num_values', \
						'average|most_presentS',
						'postitive(average|most_present)', \
						'negitive(average|most_present)', \
						'num_positive', \
						'num_negitive', 'feature_value_info'])

	fixed_str_features = np.array(load_result("str_features.csv"))[0]
	indexs = get_known_features_index(features, fixed_str_features)

	with open(file_path, "w", newline='') as csv_file:
		spamwriter = csv.writer(csv_file)
		spamwriter.writerow(first_line)

	from solve_data import feature_value_class, FeatureInData

	from collections import OrderedDict

	for fea_pos in range(1, len(features)):
		feature_info = list()
		feature_info.append(features[fea_pos])

		fea_val_cla = feature_value_class(data, fea_pos, label, fixed_str_features_index = indexs)
		feature_info.append(fea_val_cla["str_feature"])
		feature_info.append(fea_val_cla["num_of_value"])
		if fea_val_cla["str_feature"]:
			feature_info.append(fea_val_cla["most_presentS"])
			try:
				l = label[0, 0]
				feature_info.append(fea_val_cla["most_presentS_positive"])
				feature_info.append(fea_val_cla["most_presentS_negitive"])
			except:
				feature_info.append("None")
				feature_info.append("None")
		else:
			feature_info.append(fea_val_cla["average"])
			try:
				l = label[0, 0]
				feature_info.append(fea_val_cla["average_positive"])
				feature_info.append(fea_val_cla["average_negitive"])
			except:
				feature_info.append("None")
				feature_info.append("None")
		try:
			l = label[0, 0]
			feature_info.append(fea_val_cla["num_positive"])
			feature_info.append(fea_val_cla["num_negitive"])
		except:
			feature_info.append("None")
			feature_info.append("None")
		for k, v in fea_val_cla.items():
			if isinstance(v, FeatureInData):
				value_info = OrderedDict()
				value_info["value"] = k
				value_info["present_num"] = v._present_num
				try:
					l = label[0, 0]
					value_info["respond_positive_num"] = v._respond_positive_num
					value_info["respond_negitive_num"] = v._respond_negitive_num
				except:
					pass
				feature_info.append(value_info)
		with open(file_path, "a+", newline='') as csv_file:
			spamwriter = csv.writer(csv_file)
			spamwriter.writerow(feature_info)



# load the data from the file_name
#	the return will be a list which contains data [[], [], [], ...]
#	for example1: 
	# contents = load_result("data_after_delete__no_discrimination_features.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])
#	for example2:	
	# label_lines = np.array(load_result("train_label_original.csv"))
	# from save_load_result import convert_to_float
	# label = convert_to_float(label_lines)
	# code_style = "utf-8"
def load_result(file_name, dir_name = "resultData"):
	data_file_path = os.path.join(os.getcwd(), dir_name, file_name)
	
	if os.path.exists(data_file_path):
		pickle_pattern = re.compile(r".*\.pickle$")
		if pickle_pattern.search(file_name):
			print("it is a pickle")
			with open(data_file_path, "rb") as pickle_f:
				data = pickle.load(pickle_f)

		else:
			with open(data_file_path, newline='') as csv_file:
				csv_reader = csv.reader(csv_file)
				data = [line for line in csv_reader]
		return data
	else:
		print("No such file !!!")
# note: if you want to get the label data as a float type please call this function
#	-- the input: should be array types with size (row, 1)
def convert_to_int(strf_array):
	convert_result = np.array(list(map(int, strf_array[:, 0])))
	convert_result = convert_result.reshape(len(convert_result), 1)
	return convert_result


# write the deleted features into file 'all_deleted_features.csv'
# set the para re_write to control whether to re wirte or just add
# 
# note!:
#		sometimes using this function to get the feature stored in the file, maybe get a list
#	like this [[..,..,..], [..,..,..], [..,..,..]], instead we want to get features like this:
#	[.... .... ...] a array whose shape is (a, )
#		we really should do the following after we get the return from 'load_result' function
#	above -->
#		convert_to_array = np.array(load_result("all_deleted_features.csv"))
#		the_style_we_want = convert_to_array.shape((convert_to_array.size,))
def write_to_deleted_features_area(features, file_name = "all_deleted_features.csv", \
						dir_name = "resultData", re_write = False):
	file_path = os.path.join(os.getcwd(), dir_name, file_name)
	style = "a+"
	if re_write:
		style = "w"
	with open(file_path, style, newline='') as csv_file:
		spamwriter = csv.writer(csv_file)
		spamwriter.writerow(features)


def load_all_deleted_features_during_train(deleted_features_file_label = "deleted_features", \
											find_dir = "resultData/"):
	find_pattern = re.compile(r"^"+ deleted_features_file_label + ".*\.csv$")
	file_path_basis = os.path.join(os.getcwd(), find_dir)
	all_deleted_features = list()
	for file in os.listdir(file_path_basis):
		file_path = os.path.join(file_path_basis, file)
		if os.path.isfile(file_path):
			if find_pattern.search(file):
				deleted_features = load_result(file, dir_name = find_dir)
				# print(deleted_features[0])
				all_deleted_features.extend(deleted_features[0])
				# print("list: ***********")
				# print(deleted_features)
	#print(all_deleted_features)
	return all_deleted_features