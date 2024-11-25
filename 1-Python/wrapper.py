# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:58:22 2024

@author: Hp
"""

import time
def time_it(func):
    def wrapper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        end=time.time()
        total_time=(end-start)*1000
        print(func.__name__+f"took {total_time} mili sec")
        return result
    return wrapper