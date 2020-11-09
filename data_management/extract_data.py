# -*- coding: utf-8 -*-
"""Run this module from the directory where it is located! 
 
This module unzips the original data and saves them in the directory 
"build/survey_name" 

"""
import shutil
from os import listdir, rename, path
#from os import rename
#import zipfile
from pathlib import Path 
import json
from data_management.functions import unzip

__file__ = 'extract_data.py'

# COLOMBIA
# GEIH 2015 - 2019

# Unzip files 
list_years = ['2015', '2016', '2017', '2018', '2019']
src_path_list = ['../data/Colombia/GEIH_' + y + '/original_data' 
                 for y in list_years]
dst_path_list = ['../build/Colombia/GEIH_' + y + '/extracted_data' 
                 for y in list_years]

for i, y in enumerate(list_years):
    src = (Path(__file__)/src_path_list[i]).resolve()
    dst = (Path(__file__)/dst_path_list[i]).resolve()
    for o in listdir(src):
        filename = o
        unzip(filename, src, dst)

src_path_list_2 = ['../data/Colombia/GEIH_' + y 
                   for y in ['2015', '2016']] 
src_path_list_3 = ['../data/Colombia/GEIH_' + y 
                   for y in ['2017', '2018', '2019']]
dst_path_list_2 = ['../build/Colombia/GEIH_' + y 
                   for y in ['2015', '2016']]
dst_path_list_3 = ['../build/Colombia/GEIH_' + y 
                   for y in ['2017/Total_Fact_expansion', 
                             '2018/Total_Fact_expansion', 
                             '2019/Total_Fact_expansion']]

filenames_2 = ['Total Factor expansion.zip', 'Total_fact_exp_dto.csv.zip']
filenames_3 = ['Factor_expansion.zip', 'Total_Fact_expansion.zip', 
               'Total_Fact_expansion.zip']

i = 0
a = 0
for y in list_years:
    if i < 2:
        src = (Path(__file__)/
               src_path_list_2[i]).resolve()
        dst = (Path(__file__)/dst_path_list_2[i]).resolve()
        filename = filenames_2[i]
        unzip(filename,src, dst)
        i += 1
    else:
        src = (Path(__file__)/src_path_list_3[a]).resolve()
        dst = (Path(__file__)/dst_path_list_3[a]).resolve()
        filename = filenames_3[a]
        unzip(filename, src, dst)
        i += 1
        a += 1

# Change directory and file names
factor_exp_oldnames = ['../build/Colombia/GEIH_' + y 
                       for y in ['2015/Total Factor expansion', 
                                 '2016/Total_fact_exp_dto.csv', 
                                   '2017/Total_Fact_expansion', 
                                   '2018/Total_Fact_expansion', 
                                   '2019/Total_Fact_expansion']]

factor_exp_newnames = ['../build/Colombia/GEIH_' + y + '/total_factor_expansion' 
                       for y in list_years]

for i, p in enumerate(factor_exp_oldnames):
    pp = (Path(__file__)/p).resolve()
    newname = (Path(__file__)/factor_exp_newnames[i]).resolve()
    pp.rename(newname)
        
survey_parts = []
for f in listdir((Path(__file__)/'../info/Colombia/geih_variables_list').resolve()):
    if f.startswith('geih_15_area'):
        survey_parts.append(f[13:-4])

for p in dst_path_list:
    path_1 = (Path(__file__)/p).resolve()
    for d in listdir(path_1):
        dirname = path_1.joinpath(d)
        newname = Path(str(dirname)[:-4])
        dirname.rename(newname)
        path_2 = (Path(__file__)/str(dirname)[:-4]).resolve()
        a = 0
        j = 0
        l = 0
        for f in listdir(path_2):
            if f.startswith('Cabecera'):
                filename = path_2.joinpath(f)
                nn = 'cabecera_' +survey_parts[a] + filename.suffix
                newname = Path(path_2/nn)
                filename.rename(newname)
                a += 1
            elif f.startswith('Resto'):
                filename = path_2.joinpath(f)
                nn = 'resto_' +survey_parts[j] + filename.suffix
                newname = Path(path_2/nn)
                filename.rename(newname)
                j += 1
            else:
                filename = path_2.joinpath(f)
                nn = 'area_' +survey_parts[l] + filename.suffix
                newname = Path(path_2/nn)
                filename.rename(newname)
                l += 1    
                
# Delete unnecessary files
for p in factor_exp_newnames:
    pp = (Path(__file__)/p).resolve()
    for f in listdir(pp):
        if ((((Path(__file__)/pp/f).resolve()).suffix == '.SAV') | 
            (((Path(__file__)/pp/f).resolve()).suffix == '.txt') |
            (((Path(__file__)/pp/f).resolve()).suffix == '.sav') | 
            (((Path(__file__)/pp/f).resolve()).suffix == '.dta')):
            ((Path(__file__)/pp/f).resolve()).unlink()     
            
# Rename total fac exp files 
for p in factor_exp_newnames:
    pp = (Path(__file__)/p).resolve()
    for i, f in enumerate(listdir(pp)):
        filename = pp.joinpath(f)
        nn = 'total_fac_exp_' + list_years[i] + filename.suffix
        newname = Path(pp/nn)
        filename.rename(newname)

