# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:16:46 2024

@author: Hp
"""


#time calculation programm
from wrapper import time_it
@time_it
def calc_sq(numbers):
    result=[]
    for number in numbers:
        result.append(number*number)
    return result

@time_it
def calc_cube(numbers):
    result=[]
    for number in numbers:
        result.append(number*number*number)
    return result

array=range(1,1000000)
out_sq=calc_sq(array)
out_cube=calc_cube(array)





