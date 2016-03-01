#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-28 23:24:45
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os

a = '1.0'
print(int(float(a)))

import numpy as np
from save_load_result import save_result
from solve_data import convert_to_digit

data = np.array([['1','2.0','3.0','4','5'], ['6','7','32432','9','6'], ['2','3','123','51269','6234']])
print(data)
print(convert_to_digit(data))
print(int(float(data[0,1])))