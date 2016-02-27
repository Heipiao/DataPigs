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

class info_dict(OrderedDict):
	def __missing__(self, key):
		return 0
# load the csv data
def load_data(file_name, \
		data_dir = "/home/csjx/languageCode/DevelopArea/WorkAREA/Competition/PPD-First-Round-Data", \
		data_style = "Training Set"):
	data_file_path = os.path.join(data_dir, data_style, file_name)
	
	if os.path.exists(data_file_path):
		with open(data_file_path, newline='', encoding='gbk') as csv_file:
			csv_reader = csv.reader(csv_file)
			lines = [line for line in csv_reader]

		features = lines[0]
		data = lines[1:]
		original_data = lines
		return features, data, original_data
	return None

def data_info(original_data, data_label = "forTrain"):

	info = info_dict()
	info["data_label"] = data_label
	info["num_instances"] = len(original_data[1:])
	info["num_features"] = len(original_data[0][1:])

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
	if "num_target" in info:
		info["num_features"] -= 1

	return info 

def print_instance_format(features, instance_data):
	for i in range(len(features)):
		print(features[i] + ": " + instance_data[i])

if __name__ == '__main__':
	#train_master_feas, train_master_data, a = load_data("PPD_Training_Master_GBK_3_1_Training_Set.csv")
	train_master_feas, train_master_data, a = load_data("PPD_LogInfo_3_1_Training_Set.csv")
	print(train_master_feas)
	print(train_master_data[0])

	print_instance_format(train_master_feas, train_master_data[0])

	print(type(train_master_data))
	info = data_info(a)
	for k, v in info.items():
		print(k, v)

	targets = [instance[-2] for instance in train_master_data ]
	#rint(targets)