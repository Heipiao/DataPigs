#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-03-06 09:00:08
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import numpy as np


from save_load_result import save_result, load_result



if __name__ == '__main__':
	contents = load_result("after_Str_features_digited_data.csv")
	features = np.array(contents[0])
	data = np.array(contents[1:])

	from map_features_to_digit import convert_to_numerical

	data = convert_to_numerical(data, features)

