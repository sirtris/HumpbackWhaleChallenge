# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 17:32:18 2018

@author: Valentin
"""

from manip.cropping import crop_image
import pandas as pd
import os
import re

dir = os.path.dirname(__file__)
PATH = dir + "/output_crp"

should_be = pd.read_csv("IO/cropped.csv",delimiter=";")["Image"]

current = os.listdir(PATH)

i=0
tmp = []
#should_be = should_be.sample(frac=1)
for filename in should_be:
    file = filename.strip('crp_')
    i+=1
    if not os.path.isfile(PATH+"/"+filename):

        ### These are the files that are in the csv but not in any folder
        if file.find("7102d1c") != -1 or file.find("7c0893a") != -1 or file.find("f447b12_11") != -1:
            continue
#        crop_image(file)
        tmp.append(file)
        if re.search(r"_[^_|]+_",filename):
            crop_image("augmented/"+file)
        else:
            crop_image(file)