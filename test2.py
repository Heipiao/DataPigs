#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-02-27 21:04:24
# @Author  : chensijia (2350543676@qq.com)
# @Version : 0.0.0
# @Style   : Python3.5
#
# @Description: 

import os
import re

pattern_digit = re.compile(r"^[+-]?\d+")

d = "1"

if pattern_digit.match(d):
	print("is a digit")