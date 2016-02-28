#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 15:25:59
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: solve the data


from collections import defaultdict
import re
import numpy as np

# the input should be a array style
def convert_to_digit(data):
	# conver the digit number str to number
	for i in range(len(data)):
		for j in range(len(features)):
			try:
				data[i, j] = round(float(data[i][j]))
			except :
				pass
	return data

# each of the instance of this class respond to a value of the feature
#	the value of the feature is the key of value_class (see function feature_value_class)
class FeatureInData(object):
	def __init__(self):
		self._present_num = 0 #the number of this feature value present
		# this value of feature respond to the number of positive instances
		self._respond_positive_num = 0 
		self._respond_negitive_num = 0

	def present(self):
		self._present_num += 1
	def positive(self):
		self._respond_positive_num += 1
	def negitive(self):
		self._respond_negitive_num += 1

	def show(self):
		print("present_num: ", self._present_num)
		print("respond_positive_num: ", self._respond_positive_num)
		print("respond_negitive_num: ", self._respond_negitive_num)


# the input data should without first line as features` des
# the result of this function :
#	1. contain "str_feature" show whether this is a string feature
#		<2, 3> ignore the missing feature instances
#	if "str_feature" is False: 
#		2. contain "average_positive": the average value of the positive instances in this feature
#		3. contain "average_negitive"
#	if "str_feature" is True:
#		2. contain "most_presentS_positive": the most presented str of this feature (positive)
#		3. contain "most_presentS_negitive"
#	4. contain "num_positive" show how many positives instances contains except the feature is missed
#	5. contain "num_negitive" show how many negitive instances contains
#	6. contain class FeatureInData`s instances whose key is their feature value
def feature_value_class(data, fea_pos, label):
	value_class = defaultdict(FeatureInData)
	# whether this feature is a string style feature
	value_class["str_feature"] = True

	pattern_digit = re.compile(r".*\d.*")

	for i in range(len(data)):
		if data[i, fea_pos] and not data[i, fea_pos] == "不详":
			#if data[i, fea_pos] == "b"
			insert_key = data[i, fea_pos]
			try:	
				float(insert_key)
				if value_class["str_feature"]:
					value_class["str_feature"] = False
			except:
				value_class["str_feature"] = True
		else:
			insert_key = "miss"

		value_class[insert_key].present()
		if label[i, 0] == 1:
			value_class[insert_key].positive()
		else:
			value_class[insert_key].negitive()

	# for k,v in value_class.items():
		
	# 	if isinstance(v, FeatureInData):
	# 		print(k)
	# 		v.show()
	# 	else:
	# 		print("\n total: ")
	# 		print(k, v)

	max_pre_str_pos = 0
	max_pre_str_neg = 0
	num_pos = 0.0
	num_neg = 0.0
	sum_pos = 0.0
	sum_neg = 0.0
	most_presentS_positive = "None"
	most_presentS_negitive = "None"
	for k, v in value_class.items():
		#this is a string style feature
		if isinstance(v, FeatureInData):
			num_pos += v._respond_positive_num
			num_neg += v._respond_negitive_num
			if k == "miss":
				continue
			if value_class["str_feature"]:
				if v._respond_positive_num >= max_pre_str_pos:
					most_presentS_positive = k
					max_pre_str_pos = v._respond_positive_num
				if v._respond_negitive_num >= max_pre_str_neg:
					most_presentS_negitive = k
					max_pre_str_neg = v._respond_negitive_num
			else:
				sum_pos += float(k) * v._respond_positive_num
				sum_neg += float(k) * v._respond_negitive_num
	if not value_class["str_feature"]:
		value_class["average_positive"] = sum_pos / num_pos
		value_class["average_negitive"] = sum_neg / num_neg
	else:
		value_class["most_presentS_positive"] = most_presentS_positive
		value_class["most_presentS_negitive"] = most_presentS_negitive
	value_class["num_positive"] = num_pos
	value_class["num_negitive"] = num_neg
	return value_class

def label_statistics(label):
	label_sta = dict()
	positive_num = 0
	negitive_num = 0

	for i in range(len(label)):
		if label[i, 0] == 1:
			positive_num += 1
		else:
			negitive_num += 1

	label_sta["positive_num"] = positive_num
	label_sta["negitive_num"] = negitive_num

	return label_sta

# filling the miss data in the data
#	if number of miss is over threshold in one feature, just remove this features
# input:
#	- data: all the data 2-dims (a, b)
#	- features: all the features 1-dim (a,)
#	- label: the label of data 2-dims (a, b)
#	- fea_value_sat: result of function feature_value_class
def filling_miss(data, fea_pos, label, fea_value_sat, delete_fea, threshold = 5000):
	if fea_value_sat["miss"]._present_num >= threshold:
		delete_fea.append(fea_pos)
		return data

	fill_with = "fuck"
	# use the average of the same label of missed instances to fill the miss
	for i in range(len(data)):
		if not data[i, fea_pos]:
			# if this miss data is a string style
			if fea_value_sat["str_feature"]:
				if label[i, 0] == 1: # if this missed value instance`s label is 1
					fill_with = fea_value_sat["most_presentS_positive"]
				else:
					fill_with = fea_value_sat["most_presentS_negitive"]
				
			else:		
				# fill the miss with average of same label
				if label[i, 0] == 1:
					fill_with = round(fea_value_sat["average_positive"])
				else:
					fill_with = round(fea_value_sat["average_negitive"])

			#print(new_data[i, fea_pos])	
			data[i, fea_pos] = fill_with
	return data

# input:
#	fea_pos: a list contain which features should be delete
# result:
#	deleted_features: a list contain the deleted features
def delete_features(data, features, delete_fea_pos):
	deleted_features = [features[i] for i in delete_fea_pos]

	data = np.delete(data, delete_fea_pos, axis = 1)
	features = np.delete(features, delete_fea_pos, axis = 0)

	return data, features, deleted_features