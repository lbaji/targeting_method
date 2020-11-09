"""
This module contains functions used in the data management.

"""

import zipfile
from pathlib import Path 
import numpy as np
import pandas as pd
from dbfread import DBF

def unzip(filename, src, dst):
    """Unzip files.
    Args:
        filename: Full name of the file, i.e. name and extension
        src: Relaltive path to the directory where the zip file is located
        dst: Relative path to the directory where the unzipped file will be 
             located
        
    """
    zip_object = zipfile.ZipFile((Path(__file__)/src/filename).resolve(), 'r')
    zip_object.extractall((Path(__file__)/dst).resolve())
    zip_object.close()
    pass 

def add_region_column(string, df):
    """Add a new column to the dfs 'region' and add the corresponding region 
    per df.
    Args:
        string: The name of the dataframe. It is composed of the region and 
                the survey part with underscores between words
        df: The dataframe where the new column should be added.
        
    """
    region = string.split('_')[0]
    df['region'] = region
    return df 

def append_new_vars(df_base, df_other):
    """Compare one common column of two dataframes, df_base and df_other. df_other has 
    at least as many entries as df_base, normally more. Take all the rows of df_other 
    with the extra entries (if any) and append them to df_base.
    
    """
    diff_vars = np.setdiff1d(df_other['name'], df_base['name']).tolist()
    diff_vars = list(set(diff_vars))

    if len(diff_vars) > 0:
        for i in range(0, len(diff_vars)):
            to_append = df_other[df_other['name']==diff_vars[i]]
            df_base = df_base.append(to_append, ignore_index=True)
            
    return df_base

def concat_dfs(list_df_names, list_state_names, list_of_paths):
    """ Concatenate passed dataframes to create bigger ones.
    Args:
        list_df_names: list with the names of the resulting dfs. They are the keys
                    of the resulting dictionary where the resulting dfs will be
                    stored
        state_names_nac: list with the state names (Nacional included or not)
        list_of_paths: list with the paths where the data to create the 
                       auxiliary dataframes is located
        
    """
    dic_of_aux_dfs = {}
    dic_final_dfs = {}
    x = 0
    for l in list_df_names:
        dic_of_aux_dfs[l] = {k:pd.DataFrame(iter(DBF(v))) 
                            for k, v in zip(list_state_names, list_of_paths[x])}
        
        dic_final_dfs[l] = pd.concat(dic_of_aux_dfs[l])
        
        x += 1
        
    return dic_final_dfs 