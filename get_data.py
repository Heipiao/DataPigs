#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-26 23:54:21
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: get and solve the data 

import os
import csv
import re
from collections import OrderedDict
import numpy as np

import chardet

class InfoDict(OrderedDict):
	def __missing__(self, key):
		return 0
# load the csv data
def load_data(file_name, \
		data_dir = "PPD-First-Round-Data", \
		data_style = "Training Set"):

	path = os.getcwd()

	data_file_path = os.path.join(path, data_dir, data_style, file_name)
	code_style = "utf-8"
#UnicodeDecodeError
	if os.path.exists(data_file_path):
		try:
			a = open(data_file_path, newline='', encoding=code_style)
			a.read()
		except:
			print("can not open with utf-8")
			code_style = "gbk"
		
		with open(data_file_path, newline='', encoding=code_style) as csv_file:
			csv_reader = csv.reader(csv_file)
			lines = [line for line in csv_reader]

		features = lines[0]
		data = lines[1:]

		original_data = lines
		return np.array(features), np.array(data), np.array(original_data)

	print("file: " + data_file_path + "not existed")
	return None

def data_info(original_data, data_label = "forTrain"):

	info = InfoDict()
	info["data_label"] = data_label
	info["num_instances"] = len(original_data[1:])
	info["num_features"] = len(original_data[0][1:]) # ignore the idx 

	pattern1 = re.compile(r"\d")
	pattern2 = re.compile(r"\d$")
	pattern3 = re.compile(r"\d_\d")
	pattern4 = re.compile(r"_\d")
	for fea in original_data[0]:
		if not pattern1.search(fea): 
			info["num_" + fea] = 1
		if pattern2.search(fea):
			if pattern3.search(fea):
				pos = pattern3.search(fea).start()
				info["num_" + fea[0:pos]] += 1
				pos = pattern4.search(fea).start()
				info["num_" + fea[0:pos]] += 1
			elif pattern4.search(fea):
				
				pos = pattern4.search(fea).start()
				info["num_" + fea[0:pos]] += 1
			else:
				pos = pattern2.search(fea).start()			
				info["num_" + fea[0:pos]] += 1	
	# remove the target from the features			
	if "num_target" in info:
		info["num_features"] -= 1

	return info 

def print_instance_format(features, instance_data):
	for i in range(len(features)):
		print(features[i] + ": " + instance_data[i])

# the input data and features should have 'target'
# then we will extract target from them and consist the data 
#	in a new way 
def extract_target(features, data):
	if not 'target' in features:
		return features, data, None
	is_contain = features == "target"

	label = data[:, is_contain]
	not_contain_target_fea = features[is_contain == False]
	not_contain_target_data = data[:, is_contain == False]

	label = np.array(list(map(int, label[:, 0]))).reshape(len(label), 1)

	return not_contain_target_fea, not_contain_target_data, label

