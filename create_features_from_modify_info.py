#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-25 17:02:20
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import numpy as np

from save_load_result import load_result, save_result
from collections import OrderedDict

from combine_land_update_info import combine_land_modify_infos
import pickle
import pprint

import datetime


################################## create features from modify info ######################
# the structure is:
#	Users is a OrderDict() that contains all the users, its key is user`s id, 
#		its key correspond value is a OrderDict() that contains extracted features.
#		extract faetures name as a key, while result as correspond key`s value
# 	Users[user_id] = {"extracted_faetures":features_info, ....}
# the extracted features is:

#(create_features_from_whole_update_info)
#	- count_modify_frequent:	the number of whole modifyed frequent
#	- count_modify_things : the number of different value of operate code appear when modifying 
#	- most_appeared_modify_things
#	- count_most_appeared_modify_things

#(create_features_from_neared_update_info)
#	- count_modify_frequent_near_borrowed: count modify frequent 15 days near borrowed
#	- count_modify_things_near_borrowed
#   - most_appeared_modify_things_near_borrowed
#	- count_most_appeared_modify_things_near_borrowed


#(create_features_from_borrowed_update_info)
#	- count_modify_frequent_borrowed_day: meaning is just showed as the name
#	- count_modify_things_borrowed_day
#	- most_appeared_modify_things_borrowed_day
#	- count_most_appeared_modify_things_borrowed_day

#	
#(create_features_from_first_update_info)	
#	- calculate_modify_span: time between first modify and lastest modify
#	- count_modify_frequent_first_update
#	- count_first_update_modify_things: the number of different operate code value 
#	- most_apperaed_modify_things_first_update
#	- count_most_appeared_modify_things_first_update

def make_operability(user_log_info):
    borrowed_date = list(map(int, user_log_info["borrow_success_date"].split("/")))
    modified_things = [m_t[1:].lower() for m_t in user_log_info["modify_info"]["modify_things"]]
    all_modified_date = [list(map(int, date.split("/"))) for date in user_log_info["modify_info"]["modify_date"]]

    # print(borrowed_date)
    # print(modified_things)
    # print(all_modified_date)

    return borrowed_date, modified_things, all_modified_date

# ""  "" 
# "" ""  "" "" 
# "" "" "" 
# def define_update_style():
# 	phone = ["dormitoryphone", "residencephone", "secondmobile", "mobilephone", "phonetype", 
# 			"phone", "secondemail"]
# 	education = ["educationid", "schoolname", "graduateschool", "graduatedate"]
# 	money = ["incomefrom", "iscash"]
# 	third_web = ["webshopurl", "webshoptypeid", "otherwebshoptype"]
# 	private = ["marriagestatusid", "age", "hasbuycar", "realname", "qq", "relationshipid", 
# 				"gender"]
# 	id_ = ["regstepid", "orderid", "contactid", "idaddress", "userid", "idnumber", "byuserid"]
# 	others = ["hasppdaiaccount", "creationdate", "ppdaiaccount", "nickname", 
# 			"flag_uctobcp", "flag_uctopvr", "hassborgjj", "turnover", "lastupdatedate"]

# def is_location_updated(updated_thing):
# 	location = ["cityid", "position", "districtid", "residencetypeid", "residenceyears", 
# 			"residenceaddress", "provinceid", "department"]
# 	return updated_thing in location

# def is_work_updated(updated_thing):
# 	work = ["workyears", "companyname", "bussinessaddress", "companysizeid", "hasbusinesslicense",
# 		"companyaddress", "companytypeid", "companyphone"]

# 	return updated_thing in work
# def is_phone_updated(updated_thing):
# 	phone = ["dormitoryphone", "residencephone", "secondmobile", "mobilephone", "phonetype", 
# 		"phone", "secondemail"]
# 	return updated_thing in phone

# def is_education_updated(updated_thing):
# 	education = ["educationid", "schoolname", "graduateschool", "graduatedate"]

# 	return updated_thing in education
# def is_money_updated(updated_thing):
# 	money = ["incomefrom", "iscash"]
# 	return updated_thing in money
# def is_third_web_updated(updated_thing):
# 	third_web = ["webshopurl", "webshoptypeid", "otherwebshoptype"]
# 	return updated_thing in third_web
# def is_private_updated(updated_thing):
# 	private = ["marriagestatusid", "age", "hasbuycar", "realname", "qq", "relationshipid", 
# 				"gender"]

# 	return updated_thing in private
# def is_id_updated(updated_thing):
# 	id_ = ["regstepid", "orderid", "contactid", "idaddress", "userid", "idnumber", "byuserid"]

# 	return updated_thing in id_
# def is_others_updated(updated_thing):
# 	others = ["hasppdaiaccount", "creationdate", "ppdaiaccount", "nickname", 
# 			"flag_uctobcp", "flag_uctopvr", "hassborgjj", "turnover", "lastupdatedate"]

# 	return updated_thing in others

# def find_most_appeared_modify_things(modified_things):
# 	modified_thing_count = list()
# 	for thing in modified_things:
# 		if is_location_updated:
# 			modified_thing_count

def create_features_from_whole_update_info(user_log_info):
	borrowed_date, modified_things, all_modified_date = make_operability(user_log_info)

	count_modify_frequent = len(all_modified_date)
	count_modify_things = len(set(modified_things))
	
	user_update_info = OrderedDict()
	user_update_info["count_modify_frequent"] = count_modify_frequent
	user_update_info["count_modify_things"] = count_modify_things 

	return user_update_info


from create_features_from_land_operate_info import find_date_near_borrowed_day
def create_features_from_neared_update_info(user_log_info):
	borrowed_date, modified_things, all_modified_date = make_operability(user_log_info)
	near_update_date_indexs = find_date_near_borrowed_day(borrowed_date, all_modified_date)
	near_day_modified_things = [modified_things[i] for i in near_update_date_indexs]

	count_modify_frequent_near_borrowed = len(near_update_date_indexs)
	count_modify_things_near_borrowed = len(set(near_day_modified_things))

	user_near_day_update_info = OrderedDict()
	user_near_day_update_info["count_modify_frequent_near_borrowed"] = count_modify_frequent_near_borrowed
	user_near_day_update_info["count_modify_things_near_borrowed"] = count_modify_things_near_borrowed 

	return user_near_day_update_info

from create_features_from_land_operate_info import find_value_index
def create_features_from_borrowed_update_info(user_log_info):
	borrowed_date, modified_things, all_modified_date = make_operability(user_log_info)
	borrowed_date_indexs = find_value_index(all_modified_date, borrowed_date)
	borrowed_day_modified_things = [modified_things[i] for i in borrowed_date_indexs]

	count_modify_frequent_borrowed_day = len(borrowed_date_indexs)
	count_modify_things_borrowed_day = len(set(borrowed_day_modified_things))

	user_borrowed_day_update_info = OrderedDict()
	user_borrowed_day_update_info["count_modify_frequent_borrowed_day"] = count_modify_frequent_borrowed_day
	user_borrowed_day_update_info["count_modify_things_borrowed_day"] = count_modify_things_borrowed_day 

	return user_borrowed_day_update_info

from create_features_from_land_operate_info import land_date_min_max
def create_features_from_first_update_info(user_log_info):
	borrowed_date, modified_things, all_modified_date = make_operability(user_log_info)
	first_update, lastest_update, update_span = land_date_min_max(all_modified_date)
	calculate_modify_span = update_span

	first_update_date_indexs = find_value_index(all_modified_date, first_update)

	first_update_modified_things = [modified_things[i] for i in first_update_date_indexs]

	count_modify_frequent_first_update = len(first_update_date_indexs)
	count_first_update_modify_things = len(set(first_update_modified_things))

	user_first_update_info = OrderedDict()
	user_first_update_info["calculate_modify_span"] = calculate_modify_span
	user_first_update_info["count_modify_frequent_first_update"] = count_modify_frequent_first_update 
	user_first_update_info["count_first_update_modify_things"] = count_first_update_modify_things 

	return user_first_update_info


def create_features_from_update_info(combine_data):
	users_update_features = OrderedDict()
	for user in combine_data.keys():
		if not int(user) in users_update_features.keys():
			users_update_features[int(user)] = OrderedDict()

		whole_update_features = create_features_from_whole_update_info(combine_data[user])
		borrowed_update_features = create_features_from_borrowed_update_info(combine_data[user])
		first_update_features = create_features_from_first_update_info(combine_data[user])
		near_update_features = create_features_from_neared_update_info(combine_data[user])

		dictMerged1=OrderedDict(whole_update_features, **borrowed_update_features)
		dictMerged1=OrderedDict(dictMerged1, **first_update_features)
		users_update_features[int(user)]=OrderedDict(dictMerged1, **near_update_features)


	return users_update_features



if __name__ == '__main__':
	saved_dir = "resultData"
	combined_info_file = os.path.join(saved_dir, "all_id_info.pickle")
	fid=open(combined_info_file, "rb")

	data=OrderedDict(pickle.load(fid))
	fid.close()

	pprint.pprint(data["10019"])

	whole_update_features = create_features_from_whole_update_info(data["10019"])
	borrowed_update_features = create_features_from_borrowed_update_info(data["10019"])
	first_update_features = create_features_from_first_update_info(data["10019"])
	near_update_features = create_features_from_neared_update_info(data["10019"])
	


	dictMerged1=OrderedDict(whole_update_features, **borrowed_update_features)
	dictMerged1=OrderedDict(dictMerged1, **first_update_features)
	dictMerged1=OrderedDict(dictMerged1, **near_update_features)
	print(dictMerged1)
