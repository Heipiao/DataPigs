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

from solve_data import feature_value_class, FeatureInData, delete_features
from save_load_result import load_result, get_known_features_index, save_result
# view each feature`s value distributed
def view_each_features(data, features):
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	str_style_features = np.array(load_result("str_features.csv")[0])
	str_features_index = get_known_features_index(features, str_style_features)

	x = range(len(data))

	for fea_pos in range(len(features)):
		feature_name = features[fea_pos]
		# not draw the str style features
		if fea_pos in str_features_index:
			file_path = "view_data_area/csj/" + "(str" + str(fea_pos) + ")" + feature_name +  ".png"
			# print(fea_pos)
			# print(features[fea_pos])
		else:
			file_path = "view_data_area/csj/" + str(fea_pos) + ")" + feature_name +  ".png"
		y = data[:, fea_pos]
		plt.scatter(x, y)
		


		plt.xlabel("instances(30000)")
		plt.ylabel("value")
		plt.title(feature_name + " value " + "distributed " + "in instances")
		plt.ylim(-2)
		# rect = plt.bar(left = (0,1),height = (0.5,0.5),width = 0.15)
		# plt.legend((rect,),(feature_name + "`s value",))

		#print(file_path)
		plt.savefig(file_path)
		plt.close()


def view_each_features_label(data, features, label):
	data, features, deleted = delete_features(data, features, delete_feas_list=["Idx", "ListingInfo"])
	str_style_features = np.array(load_result("str_features.csv")[0])
	str_features_index = get_known_features_index(features, str_style_features)

	new_label = label.reshape((label.size,))
	x = range(len(data))
	for fea_pos in range(len(features)):
		feature_name = features[fea_pos]
		if fea_pos in str_features_index:
			file_path = "view_data_area/after_all/with_label_under_mean/" + "(str" + str(fea_pos) + ")" + feature_name +  ".png"
		else:
			file_path = "view_data_area/after_all/with_label_under_mean/" + str(fea_pos) + ")" + feature_name +  ".png"
		features_info = feature_value_class(data, fea_pos, label, str_features_index)
		if features_info["num_of_value"] > 30:
			save_result([features[fea_pos]], "complex_value_features.csv", style = "a+")
		else:
			if fea_pos not in str_features_index:
				save_result([features[fea_pos]], "simple_discrete_value_features(nonestrfeatures).csv", style = "a+")


		y_positive = data[new_label == 1, fea_pos]
		y_negitive = data[new_label == 0, fea_pos]
		positive_index = np.array([index for index in range(len(new_label)) if new_label[index] == 1])
		negitive_index = np.array([index for index in range(len(new_label)) if new_label[index] == 0])
		plt.scatter(positive_index, y_positive, marker = 'o', color = 'r', s = 10)
		plt.scatter(negitive_index, y_negitive, marker = 'x', color = 'g', s = 10)

		plt.xlabel("instances(30000)")
		plt.ylabel("value")
		if features_info["num_of_value"] < 40:
			plt.title(feature_name + " value - label " + "distributed " + "in instances" + \
						"\n the arrow --> Proportion of positive in that value & in positive")
			for k, v in features_info.items():
				if isinstance(v, FeatureInData):
					arrow_data = round(v._respond_positive_num / features_info["num_positive"] , 4)
					arrow_start_position_x = len(data) + 2000
					arrow_start_position_y = int(k)
					arrow_end_postion_x = arrow_start_position_x
					arrow_end_postion_y = int(k)
					plt.annotate(arrow_data, \
								xy=(arrow_start_position_x,arrow_start_position_y), \
								xytext=(arrow_end_postion_x,arrow_end_postion_y), \
								arrowprops=dict(facecolor='blue', shrink=0.02))

					arrow_data = round(v._respond_positive_num / v._present_num , 4)
					arrow_start_position_x = -4000
					arrow_start_position_y = int(k)
					arrow_end_postion_x = arrow_start_position_x
					arrow_end_postion_y = int(k)
					plt.annotate(arrow_data, \
								xy=(arrow_start_position_x,arrow_start_position_y), \
								xytext=(arrow_end_postion_x,arrow_end_postion_y), \
								arrowprops=dict(facecolor='blue', shrink=0.02))

		else:
			fea_average = round(np.mean(data[:, fea_pos]), 4)
			fea_std = np.std(data[:, fea_pos])
			fea_oo = round(fea_std / fea_average, 4)
			max_v = np.amax(data[:, fea_pos])
			min_v = np.amin(data[:, fea_pos])
			plt.title(feature_name + " | mean & Proportion of positive under that mean" + \
				"\n degree of fluctuation --> " + str(fea_oo))
			x1 = np.array(range(-5000, 35000))
			y_mean = fea_average * np.ones((x1.size))
			#plt.plot(x1, y_mean, color = 'k', linestyle = "--")
			plt.annotate(fea_average, \
								xy=(-4000,fea_average), \
								xytext=(-4000,fea_average), \
								arrowprops=dict(facecolor='blue', shrink=0.05))
			under_mean_positive = 0
			under_mean_num = 0

			for k, v in features_info.items():
				if isinstance(v, FeatureInData):
					if k <= fea_average:
						under_mean_num += v._present_num
						under_mean_positive += v._respond_positive_num
			ave_posi = round(under_mean_positive / features_info["num_positive"], 4)
			plt.annotate(ave_posi, \
					xy=(31000,fea_average), \
					xytext=(31000,fea_average), \
					arrowprops=dict(facecolor='blue', shrink=0.05))
			pos_rat = 0
			pos_rat_whole = 0
			if -1 in features_info.keys():
				pos_rat = features_info[-1]._respond_positive_num / features_info[-1]._present_num
				pos_rat_whole = features_info[-1]._respond_positive_num / features_info["num_positive"]
				plt.annotate(round(pos_rat_whole, 4), \
						xy=(31000,-1), \
						xytext=(31000,-1))
				plt.annotate(round(pos_rat, 4), \
						xy=(-4000,-1), \
						xytext=(-4000,-1))
			plt.ylim(min_v - 10, fea_average * 2)
			#plt.ylim(fea_average - round(fea_average / 10), max_v + round(fea_average / 10))
		plt.savefig(file_path)
		plt.close()

# if set draw_values --> just draw a part of fea values
#	draw_values should be a list show which values to draw as [value1 value2 ...]
def draw_feature_value_num(fea_info, draw_values = " ", \
							file_name = "value.png"):

	x = draw_values
	if draw_values == " ":
		x = [k for k in fea_info.keys() if isinstance(fea_info[k], FeatureInData)]

	y = [fea_info[fea_value]._present_num for fea_value in draw_values]
	plt.bar(x, y, color = 'g')

	plt.savefig(file_name)
	plt.close()

def draw_features_value_label_rate(fea_info, draw_values = " ", \
							file_name = "value.png"):

	x = draw_values
	if draw_values == " ":
		x = [k for k in fea_info.keys() if isinstance(fea_info[k], FeatureInData)]


	y_positive = [round(fea_info[fea_value]._respond_positive_num / fea_info[fea_value]._present_num, 4) \
					for fea_value in x]
	y_negitive = [round(fea_info[fea_value]._respond_negitive_num / fea_info[fea_value]._present_num, 4) \
					for fea_value in x]
	plt.plot(x, y_positive, color = 'r')
	plt.plot(x, y_negitive, '--', color = 'g')

	plt.savefig(file_name)
	plt.close()

def draw_features_value_weight(fea_info, draw_values = " ", \
							file_name = "value.png"):

	x = draw_values
	if draw_values == " ":
		x = [k for k in fea_info.keys() if isinstance(fea_info[k], FeatureInData)]
	
	def weight_positive(fea_value):
		return (fea_info[fea_value]._respond_positive_num / fea_info[fea_value]._present_num) * \
				(fea_info[fea_value]._respond_positive_num / fea_info["num_positive"])

	def weight_negitive(fea_value):
		return (fea_info[fea_value]._respond_negitive_num / fea_info[fea_value]._present_num) * \
				(fea_info[fea_value]._respond_negitive_num / fea_info["num_negitive"])

	y1 = [round(weight_positive(fea_value), 4) for fea_value in draw_values]
	y2 = [round(weight_negitive(fea_value), 4) for fea_value in draw_values]

	plt.plot(x, y1, color = 'r')
	plt.plot(x, y2, '--', color = 'g')
	plt.ylim(-0.001, 0.1)
	plt.savefig(file_name)
	plt.close()
## fea_respond_data should be data[:, fea_pos]
## fea_info should be the return of function 'feature_value_class'
## view_precise: (default) means how many values draw in one pic
# output:
#	.png style files
def view_feature_value_info(fea_respond_data, fea_info, saved_area, view_precise = 200):
	values_in_fea = np.array(list(set(list(fea_respond_data))))
	sorted_fea_data = values_in_fea[np.argsort(values_in_fea)]

	draw_button_edge_index = 0
	draw_upper_edge_index = view_precise

	if draw_upper_edge_index >= len(sorted_fea_data):
		draw_upper_edge_index = len(sorted_fea_data)

	part = 0
	while draw_upper_edge_index < len(sorted_fea_data) + 1:
		draw_for_each = list()
		draw_values = sorted_fea_data[draw_button_edge_index:draw_upper_edge_index]

		part += 1
		file_name = os.path.join(saved_area, "valuePart" + str(part) + ".png")
		# draw_features_value_label_rate(fea_info, draw_values, file_name)
		draw_features_value_weight(fea_info, draw_values, file_name)
		if draw_upper_edge_index >= len(sorted_fea_data):
			break

		draw_button_edge_index = draw_upper_edge_index
		draw_upper_edge_index += view_precise
		if draw_upper_edge_index >= len(sorted_fea_data):
			draw_upper_edge_index = len(sorted_fea_data)

import re

def view_tp_cv(type_info, sat_type, sat_name, saved_fig = "view_res.png"):
	res = [v[sat_type][sat_name] for v in type_info.values()]
	x = range(0, len(res))
	plt.scatter(x, res, s = 10)

	ave_res = [value for value in res if not value == 0]
	plt.title("average all the data showed below: " + str(round(float(np.mean(ave_res)), 1)))
	plt.savefig(saved_fig)
	plt.close()


def store_pic_with_class(pic_dir = "view_data_area/csj"):

	def copy_to_dir(file, dir_name):
		pass

	pic_class = ["UserInfo", "Education_Info"]
	dir_path = os.path.join(os.getcwd(), pic_dir)
	find_pattern = re.compile(r".*\.png$")

	for file in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file)
		if os.path.isfile(file_path):
			if find_pattern.search(file):
				pass
	return all_deleted_features	


import shutil

def store_thirdPart_info_by_class(pic_dir = "view_data_area/csj"):
	dir_path = os.path.join(os.getcwd(), pic_dir)
	find_pattern = re.compile(r".*\.png$")
	for file in os.listdir(dir_path):
		file_path = os.path.join(dir_path, file)
		if os.path.isfile(file_path):
			if find_pattern.search(file):
				# just operate here, we can use store the png files departement
				name = file.split("_")
				tp_class = name[3].split(".")[0]
				#######################################
				saved_dir = os.path.join(dir_path, tp_class)
				if not os.path.exists(saved_dir):
					os.mkdir(saved_dir)

				if os.path.isdir(saved_dir):
					shutil.copy(file_path, saved_dir)

	print("Copy Done")

if __name__ == '__main__':
	
	# contents = load_result("data_of_ThirdPart.csv")
	# features = np.array(contents[0])
	# data = np.array(contents[1:])

	# label_lines = np.array(load_result("train_label_original.csv"))
	# from save_load_result import convert_to_float
	# label = convert_to_float(label_lines)

	# from map_features_to_digit import convert_to_numerical

	# data = convert_to_numerical(data, features)
	#view_each_features(data, features)

	store_thirdPart_info_by_class(pic_dir = "view_data_area/after_all/with_label_under_mean/")


	# #print(np.where(data[:, 1] == -1))
	#view_each_features_label(data, features, label)


	# # file_dir = "view_data_area/view_features_weight/"
	# # str_style_features = np.array(load_result("str_features.csv")[0])
	# # str_features_index = get_known_features_index(features, str_style_features)

	# # for fea_pos in range(1, len(features)):
	# # 	if fea_pos not in str_features_index:
	# # 		fea_info = feature_value_class(data, fea_pos, label, str_features_index)
	# # 		if fea_info["num_of_value"] > 30:
	# # 			os.mkdir(os.path.join(file_dir, features[fea_pos]))
	# # 			saved_area = os.path.join(file_dir, features[fea_pos])			
	# # 			view_feature_value_info(data[:, fea_pos], fea_info, saved_area)