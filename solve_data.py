#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 15:25:59
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: solve the data

from collections import defaultdict, OrderedDict
import re
import numpy as np




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
#	fea_pos --> not contain 0, because this respond features is Id
#	fixed_str_features_index --> a list contain the index of str style features
#		why use this ? because even if the str style features are digited, 
#			we really should solve these features as str
#	you should use the sentence below to get the fixed_str_features_index:
		# 	fixed_str_features = np.array(load_result("str_features.csv")[0])
		#	indexs = list()
		#	for i in range(len(fixed_str_features)):
		#	try:
		#		finded = np.where(features == fixed_str_features[i])[0][0]
		#		indexs.append(finded)
		#	except:
		#		pass
		#	fixed_str_features_index = indexs
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
def feature_value_class(data, fea_pos, label, fixed_str_features_index = " "):
	value_class = defaultdict(FeatureInData)
	# whether this feature is a string style feature
	value_class["str_feature"] = True
	for i in range(len(data)):
		if not data[i, fea_pos] == " " and not data[i, fea_pos] == "":
			# finally i think we can not solve '不详'
			#	so i add the replace the col contain '不详'
			#	with the original one
			if data[i, fea_pos] == "不详" or str(data[i, fea_pos]) == "-1":
				insert_key = -1
				#print(data[i, fea_pos])
			else:
				insert_key = data[i, fea_pos]
				try:	
					float(insert_key)
					if value_class["str_feature"]:
						value_class["str_feature"] = False
				except:
					value_class["str_feature"] = True
				if not fixed_str_features_index == " ":
					if fea_pos in fixed_str_features_index:
						value_class["str_feature"] = True
		else:
			insert_key = -1
			#print(data[i, fea_pos])
		value_class[insert_key].present()
		try:
			if label[i, 0] == 1:
				value_class[insert_key].positive()
			else:
				value_class[insert_key].negitive()
		except:
			pass

	num_of_value = 0
	max_pre_str_pos = 0
	max_pre_str_neg = 0
	max_pre_str = 0
	num_pos = 0.0
	num_neg = 0.0
	sum_pos = 0.0
	sum_neg = 0.0
	total_sum = 0.0
	most_presentS_positive = "None"
	most_presentS_negitive = "None"
	most_presentS = "None"
	for k, v in value_class.items():
		#this is a string style feature
		if isinstance(v, FeatureInData):
			num_of_value += 1
			try:
				l = label[0, 0]
				num_pos += v._respond_positive_num
				num_neg += v._respond_negitive_num
			except:
				pass
			if k == -1:
				continue
			if value_class["str_feature"]:
				# get the most present str value
				if v._present_num >= max_pre_str:
					max_pre_str = v._present_num
					most_presentS = k
				# if we have label, get more imformation
				try:
					l = label[0, 0]
					if v._respond_positive_num >= max_pre_str_pos:
						most_presentS_positive = k
						max_pre_str_pos = v._respond_positive_num
					if v._respond_negitive_num >= max_pre_str_neg:
						most_presentS_negitive = k
						max_pre_str_neg = v._respond_negitive_num
				except:
					pass
			else:
				total_sum += float(k) * v._present_num
				try:
					l = label[0, 0]
					sum_pos += float(k) * v._respond_positive_num
					sum_neg += float(k) * v._respond_negitive_num
				except:
					pass
	value_class["num_of_value"] = num_of_value
	if not value_class["str_feature"]:
		value_class["average"] = round(total_sum / len(data))
		try:
			l = label[0, 0]
			value_class["average_positive"] = round(sum_pos / num_pos, 3)
			value_class["average_negitive"] = round(sum_neg / num_neg, 3)
		except:
			pass
	else:
		value_class["most_presentS"] = most_presentS
		try:
			l = label[0, 0]
			value_class["most_presentS_positive"] = most_presentS_positive
			value_class["most_presentS_negitive"] = most_presentS_negitive
		except:
			pass
	try:
		l = label[0, 0]
		value_class["num_positive"] = num_pos
		value_class["num_negitive"] = num_neg
	except:
		pass

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
# ?? how to use this funcion? --> Please see the main.py
def replace_miss_with_specialV(data, fea_pos, fea_value_sat, label,delete_fea, missing_num, threshold = 29000):
	try:
		l = label[0, 0]
		if fea_value_sat["-1"]._present_num >= threshold:
			delete_fea.append(fea_pos)
			missing_num.append(fea_value_sat["-1"]._present_num)
			return data
	except:
		pass
	fill_with = "fuck"
	# if label has: 
	# use the average of the same label of missed instances to fill the miss
	for i in range(len(data)):
		if not data[i, fea_pos] or data[i, fea_pos] == "不详" or data[i, fea_pos] == "-1":
			# if this miss data is a string style
			# if fea_value_sat["str_feature"]:
			# 	fill_with = fea_value_sat["most_presentS"]
			# 	try:
			# 		if label[i, 0] == 1: # if this missed value instance`s label is 1
			# 			fill_with = fea_value_sat["most_presentS_positive"]
			# 		else:
			# 			fill_with = fea_value_sat["most_presentS_negitive"]
			# 	except:
			# 		pass		
			# else:		
			# 	fill_with = fea_value_sat["average"]
			# 	# fill the miss with average of same label, if label has
			# 	try:
			# 		if label[i, 0] == 1:
			# 			fill_with = round(fea_value_sat["average_positive"])
			# 		else:
			# 			fill_with = round(fea_value_sat["average_negitive"])
			# 	except:
			# 		pass
			#print(new_data[i, fea_pos])	
			fill_with = -1
			data[i, fea_pos] = str(fill_with)

	return data

def filling_miss_with_experience(data, features, fea_pos):
	experient_features_content = load_result("experience_features_info.csv")
	experient_data_label = np.array(experient_features_content[0])
	experient_features_data = np.array(experient_features_content[:])
	now_feature_name = features[fea_pos]

	which_row = np.where(experient_features_data[:, 0] == now_feature_name)[0][0]
	which_col = np.where(experient_data_label == "average|most_presntS")

	fill_with = experient_features_data[which_row, which_col]
	for i in range(len(data)):
		if not data[i, fea_pos] or data[i, fea_pos] == "不详" or data[i, fea_pos] == "-1":
			data[i, fea_pos] = fill_with
	return data

# input:
#	delete_fea_pos: a list contain which features should be delete
#	delete_feas_list: a default para contain the name of features you want to remove
# result:
#	deleted_features: a list contain the deleted features` index
def delete_features(data, features, delete_fea_pos=None, delete_feas_list=None):
	deleted_features = []
	if not delete_fea_pos == None:
		deleted_features = [features[i] for i in delete_fea_pos]

	if not delete_feas_list == None:
		deleted_features = delete_feas_list
		delete_fea_pos = list()
		for f in delete_feas_list:
			try:
				fea_pos = np.where(features == f)[0][0]
				delete_fea_pos.append(fea_pos)
			except:
				pass


	data = np.delete(data, delete_fea_pos, axis = 1)
	features = np.delete(features, delete_fea_pos, axis = 0)

	return data, features, deleted_features



# input:
#	fea_value_sat: should be the result of the function 'feature_value_class'
def calculate_feature_entroy(fea_value_sat):
	from math import log
	log2 = lambda x : log(x) / log(2)


	num_pos = fea_value_sat["num_positive"]
	num_neg = fea_value_sat["num_negitive"]
	if num_pos == 0 or num_neg == 0:
		return 0
	pos_entroy = 0.0
	neg_entroy = 0.0
	num_instances = num_pos + num_neg
	for k, v in fea_value_sat.items():
		if isinstance(v, FeatureInData):
			pro_pos = float(v._respond_positive_num / num_pos)
			pro_neg = float(v._respond_negitive_num / num_neg)
			if not pro_pos == 0:
				pos_entroy = pos_entroy - pro_pos * log2(pro_pos)
			if not pro_neg == 0:
				neg_entroy = neg_entroy - pro_neg * log2(pro_neg)

	fea_entroy =  float(num_pos / num_instances) * pos_entroy + \
				float(num_neg / num_instances) * neg_entroy
	fea_entroy = round(fea_entroy, 2)	
	return fea_entroy

## I think the method to calcualte entroy defined in function 'calculate_feature_entroy' is wrong
#	so i write a new one here
def calculate_feature_entroy_new(fea_value_sat, total_instances_num):
	from math import log
	log2 = lambda x : log(x) / log(2)
	num_of_value = fea_value_sat["num_of_value"]
	if num_of_value == 1:
		return 0
	num_pos = fea_value_sat["num_positive"]
	num_neg = fea_value_sat["num_negitive"]
	if num_pos == 0 or num_neg == 0:
		return 0



	value_pre_entroy = calculate_feature_entroy(fea_value_sat)
	threshold_pos_rat = 0.0733 # in the whole data set, the ratio of positive number

	for k, v in fea_value_sat.items():
		if isinstance(v, FeatureInData):
			# pro_pre = float(v._present_num / total_instances_num)
			#value_pre_entroy = value_pre_entroy - pro_pre * log2(pro_pre)

			# this value contain too much instances and its useful
			if (v._respond_positive_num / v._present_num) > threshold_pos_rat \
				and v._present_num > (total_instances_num / fea_value_sat["num_of_value"]):
				value_pre_entroy += 1# / fea_value_sat["num_of_value"]

			if v._present_num > 20000 and (v._respond_positive_num / v._present_num) < threshold_pos_rat:
				value_pre_entroy -= 4
			

	return round(value_pre_entroy,4)
# calculate all the entroy of each feature and sort them in order lower --> bigger
#	contain all the features` entroy in a dict 
#		- key: index of the feature
#		- value: entroy of the respond feature
def sort_features_with_entroy(data, features, label):
	index_entroy = dict()
	for fea_pos in range(1, len(features)):
		fea_value_sat = feature_value_class(data, fea_pos, label)
		fea_pos_entroy = calculate_feature_entroy_new(fea_value_sat, len(data))
		index_entroy[fea_pos] = fea_pos_entroy
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	x = np.array(list(index_entroy.keys()))
	y = np.array(list(index_entroy.values()))
	plt.scatter(x, y)
	plt.xlabel("index of features(1-->total)")
	plt.ylabel("measure for each")

	plt.savefig("test_show.png")
	plt.close()
	temp = sorted(index_entroy.items(), key=lambda d: d[1])
	sorted_index_entroy = OrderedDict()
	for i in range(len(temp)):
		sorted_index_entroy[temp[i][0]] = temp[i][1]
	return sorted_index_entroy
# after filling the missing features and remove the most missing contained features
#	now we remove the features without enough discrimination
# input:
#	- index_entroy: the return result of function 'sort_features_with_entroy'
#	- lower_number_deleted: the number of lowest entroy features you want to delete
# output:
#	- the three return just as function 'delete_features'
#	- a list contain the responding entroy of the deleted features
def delete_no_discrimination_features(data, features, index_entroy, lower_number_deleted = " "):
	delete_fea_index = []
	delete_fea_entroy = []
	number = 0
	for k, v in index_entroy.items():
		if not lower_number_deleted == " ":
			if number <= lower_number_deleted:
				delete_fea_index.append(k)
				delete_fea_entroy.append(v)
				number += 1
		else:
			if v <= 4:
				delete_fea_index.append(k)
				delete_fea_entroy.append(v)
	# delete_fea_index = np.array(delete_fea_index)
	# delete_fea_entroy = np.array(delete_fea_entroy)
	data, features, deleted_features = delete_features(data, features, delete_fea_index)
	return data, features, deleted_features, delete_fea_entroy


# input:
#	features: the features from outside
#	known_features: the features you want to know the index in features
def get_known_features_index(features, known_features):

	contain_respond_features_index = list()
	for i in range(len(known_features)):
		try:
			finded_index = np.where( features == known_features[i])[0][0]
			contain_respond_features_index.append(finded_index)
		except:
			pass
	return contain_respond_features_index





def fill_the_missing_after_all(data, fea_pos, fea_value_sat, label = None):

	# if label has: 
	# use the average of the same label of missed instances to fill the miss
	for i in range(len(data)):
		if data[i, fea_pos] == -1:
			# if this miss data is a string style
			if fea_value_sat["str_feature"]:
				fill_with = fea_value_sat["most_presentS"]
				try:
					if label[i, 0] == 1: # if this missed value instance`s label is 1
						fill_with = fea_value_sat["most_presentS_positive"]
					else:
						fill_with = fea_value_sat["most_presentS_negitive"]
				except:
					pass		
			else:		
				fill_with = fea_value_sat["average"]
				# fill the miss with average of same label, if label has
				try:
					if label[i, 0] == 1:
						fill_with = round(fea_value_sat["average_positive"])
					else:
						fill_with = round(fea_value_sat["average_negitive"])
				except:
					pass
			data[i, fea_pos] = fill_with
	return data
# this function is aim to extract the instances with too many missing...
#### want to use these person who have lots of missing information to
#### create a module to description these person who are lack of information

if __name__ == '__main__':
	from save_load_result import convert_to_float, load_result
	label_lines = np.array(load_result("train_label_original.csv"))
	print(label_lines.shape)

	label = convert_to_float(label_lines)
	label_sta = label_statistics(label)
	print(label_sta)
	posi = label_sta["positive_num"]
	negi = label_sta["negitive_num"]
	threshold1 = round(posi / negi, 4)
	threshold2 = round(posi / (posi + negi), 4)
	print(threshold1) # 0.0791
	print(threshold2) # 0.0733