# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:38:57 2020

@author: laura
"""

#import pandas as pd
import numpy as np
from os import listdir
from pathlib import Path 
#from dbfread import DBF
#import json
#from data_management.functions import add_region_column, append_new_vars
import pickle
import yaml

__file__ = 'create_dfs_2.py'

# COLOMBIA
# GEIH 2015 - 2019

# Load dfs 
survey_parts = []
for f in listdir((Path(__file__)/'../info/Colombia/geih_variables_list').resolve()):
    if f.startswith('geih_15_area'):
        survey_parts.append(f[13:-4])

geih_parts = {}
var_info = {}
for part in survey_parts:
    path_1 = '../build/Colombia/output_dfs/' + part + '.pickle'
    path_2 = '../build/Colombia/output_dfs/var_info_' + part + '.pickle'
    p_in_1 = open((Path(__file__)/path_1).resolve(), 'rb')
    p_in_2 = open((Path(__file__)/path_2).resolve(), 'rb')
    geih_parts[part] = pickle.load(p_in_1)
    var_info[part] = pickle.load(p_in_2)
    
# Encode dfs' column values with the yaml files
dict_encoding = {}
for part in survey_parts:
    path = '../info/Colombia/geih_encoding/' + part + '.yaml'
    with open((Path(__file__)/path).resolve(), 'r', encoding='utf8') as file:
        dict_encoding[part] = yaml.load(file, Loader=yaml.FullLoader)

for part in survey_parts:
    geih_parts[part].replace(dict_encoding[part], inplace=True)
    
# 'np.nan' from yaml files must be converted to np.nan, cannot pass it without 
# the "" 
for part in survey_parts:
    geih_parts[part].replace({'np.nan': np.nan}, inplace=True) 
    
# DTYPES
dict_dtypes = {}
for part in survey_parts:
    dict_dtypes[part] = {k:v for k, v in 
           zip(var_info[part]['name'], var_info[part]['new_dtype'])}
    
# Change dtypes 
for part in survey_parts:
    geih_parts[part] = geih_parts[part].astype(dict_dtypes[part])

# Replace 'nan' with np.nan in column p1661s4a1'/'other_last_12_m_amount' in 
# df 'otros_ingresos'
#for part in survey_parts:
#    geih_parts[part].replace({'nan': np.nan}, inplace=True)
    
geih_parts['otros_ingresos']['p1661s4a1'].replace({'nan': np.nan}, inplace=True)

# Rename dfs columns 
dict_newnames = {}
for part in survey_parts:
    dict_newnames[part] = {k:v for k, v in zip(var_info[part]['name'], 
                                               var_info[part]['new_name'])}

for part in survey_parts:
    geih_parts[part].rename(columns=dict_newnames[part], inplace=True)
    
# Include info from 'area_city' variable and then drop area duplicates and save 
# output dataframes
for part in survey_parts:
    if part == 'vivienda_y_hogares':
        duplicates = geih_parts[part][
        geih_parts[part]['region']!='resto'].duplicated(
        subset=['directorio', 'secuencia_p', 'household'])
    else:        
        duplicates = geih_parts[part][
            geih_parts[part]['region']!='resto'].duplicated(
            subset=['directorio', 'secuencia_p', 'orden', 'household'])              
    duplicates_city = duplicates.loc[duplicates==False]
    duplicates_no_city = duplicates.loc[duplicates==True]
    union = geih_parts[part].loc[duplicates_city.index]['directorio'].isin(
        geih_parts[part].loc[duplicates_no_city.index]['directorio'])
    duplicates_city = union.loc[union==True]

    for i, index in enumerate(duplicates_city.index):
        geih_parts[part].loc[
            duplicates_no_city.index[i], 'area_city'] = geih_parts[part].loc[
            index, 'area_city']
                
    if part == 'vivienda_y_hogares':
        geih_parts[part].drop_duplicates(
            ['directorio', 'secuencia_p', 'household'], keep='last', 
            inplace=True, ignore_index=True)
    else:
        geih_parts[part].drop_duplicates(
            ['directorio', 'secuencia_p', 'orden', 'household'], keep='last', 
            inplace=True, ignore_index=True)
    path_1 = '../build/Colombia/output_dfs/' + part + '_final.pickle'
    geih_parts[part].to_pickle((Path(__file__)/path_1).resolve()) 