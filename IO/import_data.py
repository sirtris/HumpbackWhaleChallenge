# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:17:24 2018

@author: Valentin
"""

import pandas as pd
#import os
#import re

def import_training_data():
    return pd.read_pickle('IO/training_input.gz')
    
    


#cropped = pd.read_csv("cropped.csv",delimiter=";")
#filenames = cropped["Image"].str.extract('^[^_]+_([^_|.]+)(_|.)[^_]+$')[0]
#
#cropped['Image'] = 'cropped/' + cropped['Image'].astype(str)
#cropped['key'] = filenames
#       
#orig = pd.read_csv('original_features.csv',delimiter=";")
#orig = orig.drop(['Unnamed: 7'],axis=1)
#filenames = orig["id"].str.extract('([^_|.]+)(_|.)[^_]+$')[0]
#orig['key'] = filenames
#    
#data = pd.merge(cropped,orig,on='key').drop(['id'],axis=1)
#data.to_pickle('training_input.gz', compression='gzip')

#data = pd.read_pickle('training_input.gz')

#data = import_data()
#print(data.head())

