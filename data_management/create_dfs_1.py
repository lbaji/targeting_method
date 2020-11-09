"""Run this module from the directory where it is located!

This module creates dfs for each survey from the extracted data (see module 
'extract_data.py'). The new column names and dtypes of the dfs are defined with 
json files in the directory 'info/'. 

Missing values are set to np.nan.

"""
from os import listdir
import pandas as pd
from pathlib import Path 
from data_management.functions import add_region_column, append_new_vars

__file__ = 'create_dfs_1.py'

# COLOMBIA
# GEIH 2015 - 2019

# Create dfs for each survey part across months
list_years = ['2015', '2016', '2017', '2018', '2019']
dst_path_list = ['../build/Colombia/GEIH_' + y + '/extracted_data' 
                 for y in list_years]
list_months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 
               'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
list_months_17_csv = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                      'Julio', 'Agosto']
list_months_17_txt = ['Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

list_df_names = []
p = '../build/Colombia/GEIH_2015/extracted_data/Abril'
for f in listdir((Path(__file__)/p).resolve()):
    list_df_names.append(f)

list_df_names_17 = []
p = '../build/Colombia/GEIH_2017/extracted_data/Septiembre'
for f in listdir((Path(__file__)/p).resolve()):
    list_df_names_17.append(f)

geih_dict_monthly = {}
for i, p in enumerate(dst_path_list):
    if i == 2:
        geih_dict_monthly['2017'] = {n[:-4]:{m:pd.read_csv(
            (Path(__file__)/p/m/n).resolve(), sep=';',encoding='iso8859_15',
            low_memory=False) for m in list_months_17_csv} 
            for n in list_df_names}
        
        for n in list_df_names_17:
            for m in list_months_17_txt:
                geih_dict_monthly['2017'][n[:-4]][m] = pd.read_csv(
                 (Path(__file__)/p/m/n).resolve(),sep='\t', 
                 encoding='iso8859_15', low_memory=False) 
                
    else:
        geih_dict_monthly[list_years[i]] = {n[:-4]:{m:pd.read_csv(
            (Path(__file__)/p/m/n).resolve(), sep=';', encoding='iso8859_15', 
            low_memory=False) for m in list_months} for n in list_df_names}
   
# Change dfs' column names to lower-cases and change column name 'area' in the 
# area and cabecera dfs to avoid confusion
for y in list_years:
    for k_part in geih_dict_monthly[y].keys():
        for k_month in geih_dict_monthly[y][k_part].keys():
            geih_dict_monthly[y][k_part][k_month].columns = map(str.lower, 
                            geih_dict_monthly[y][k_part][k_month].columns)
            if k_part.startswith('area'):
                geih_dict_monthly[y][k_part][k_month].rename(
                                   columns={'area': 'area_city'}, inplace=True)

            elif k_part.startswith('cabecera'):
                geih_dict_monthly[y][k_part][k_month].rename(
                              columns={'area':'area_department'}, inplace=True)

            else:
                geih_dict_monthly[y][k_part][k_month].rename(
                                   columns={'area':'area_resto'}, inplace=True)
                
            geih_dict_monthly[y][k_part][k_month].rename(
                                        columns={'ï»¿directorio': 'directorio', 
                                              'ï»¿secuencia_p': 'secuencia_p'}, 
                                                                  inplace=True)

# Create dfs for each survey part across years
geih_dict_yearly = {}
for y in list_years: 
    geih_dict_yearly[y] = {n[:-4]:pd.concat(geih_dict_monthly[y][n[:-4]]) 
                           for n in list_df_names}
    
# Drop error columns (not in documentation and not in other years or 
# survey_parts, !!from problematic 2017!!)
geih_dict_yearly['2017']['area_ocupados'].drop(columns=['p6360', 'p6360s1'], 
                                               inplace=True)
geih_dict_yearly['2017']['area_otras_actividades_y_ayudas_en_la_semana'].drop(
    columns=['p7481s1', 'p7481s2', 'p7481s3', 'p7481s4', 'p7481s5', 'p7481s6', 
                      'p7481s7', 'p7481s8', 'p7482', 'p7482s1'], inplace=True)

# Concatenate dfs over years adding one column with the year and another one 
# with the month. Change from multi-index to single-level index
for y in list_years:
    for key in geih_dict_yearly[y].keys():
        geih_dict_yearly[y][key]['year'] = int(y)
        geih_dict_yearly[y][key].reset_index(level=0, inplace=True)
        geih_dict_yearly[y][key].rename(columns={'level_0':'month_letters'}, 
                                        inplace=True)
        
lists_part_by_region = [[] for key in geih_dict_yearly['2015'].keys()]
for y in list_years:
    for i, key in enumerate(geih_dict_yearly[y].keys()):
        lists_part_by_region[i].append(geih_dict_yearly[y][key])
        
geih_part_by_region= {}
for i, key in enumerate(geih_dict_yearly['2015'].keys()):
    geih_part_by_region[key] = pd.concat(lists_part_by_region[i], 
                                         ignore_index=True)
    
# Concatenate dfs over regions adding one column with the region
for key in geih_part_by_region.keys():
    geih_part_by_region[key] = add_region_column(key, geih_part_by_region[key])
 
survey_parts = []
for f in listdir((Path(__file__)/'../info/Colombia/geih_variables_list').resolve()):
    if f.startswith('geih_15_area'):
        survey_parts.append(f[13:-4])    
    
regions = ['area', 'cabecera', 'resto']

lists_parts = [[] for part in survey_parts]
for region in regions:
    for i, part in enumerate(survey_parts):
        key = region + '_' + part 
        lists_parts[i].append(geih_part_by_region[key])
        
geih_parts = {part:pd.concat(lists_parts[i], ignore_index=True) 
              for i, part in enumerate(survey_parts)}

# Set values different than 1 and 2 ('yes' and 'no') to 1 for categorical 
# variable 'p6030' ('replied_birth_date'). These values are the date 
geih_parts['caracteristicas_generales_personas'].loc[
    (geih_parts['caracteristicas_generales_personas']['p6030'].notna()) &
    (geih_parts['caracteristicas_generales_personas']['p6030']!=1) & 
    (geih_parts['caracteristicas_generales_personas']['p6030']!=2),'p6030'] = 1

# Create dfs with variable info (names, dtypes) and store them in a dictionary
variable_list_dic = {}
p = '../info/Colombia/geih_variables_list'
list_y = ['15', '16', '17', '18', '19']
list_prefixes = ['geih_' + y + '_' for y in list_y]
for part in survey_parts: 
    variable_list_dic[part] = {region:{y[5:-1]:pd.read_csv(
        (Path(__file__)/p/str(y+region+'_'+part+'.csv')).resolve(), sep=';', 
        encoding='iso8859_15') for y in list_prefixes} for region in regions}

# Change dfs' column names and first column ('name') entries to lower-cases and 
# drop rows with missings in 'name' 
for part in survey_parts:
    for region in regions:
        for y in list_y:
            variable_list_dic[part][region][y].columns = map(str.lower, 
                                    variable_list_dic[part][region][y].columns)
            a = variable_list_dic[part][region][y]['name']
            variable_list_dic[part][region][y]['name'] = a.str.lower()
            variable_list_dic[part][region][y].dropna(subset=['name'], 
                                                      inplace=True)
            
# Change variable name 'area' in the area and cabecera dfs to avoid confusion
for part in survey_parts:
    for region in regions:
        for y in list_y:
            if region == 'area':
                variable_list_dic[part]['area'][y].replace(
                {'name': {'area': 'area_city'}, 
                 'new_name': {'area': 'area_city'}}, inplace=True)
            elif region == 'cabecera':
                variable_list_dic[part]['cabecera'][y].replace(
                       {'name': {'area': 'area_department'}, 
                        'new_name': {'area': 'area_department'}}, inplace=True)
                
            else:
                variable_list_dic[part]['resto'][y].replace(
                            {'name': {'area': 'area_resto'},
                             'new_name': {'area': 'area_resto'}}, inplace=True)
                
# Find all new variables added over the years to each survey part for each 
# region and append them to the 2015's df for each survey part and region 
# (append over years)
for part in survey_parts:
    for region in regions:
        for i in range(0, len(list_y)):
            if i > 0:
                df_other = variable_list_dic[part][region][list_y[i]]
                df_base = variable_list_dic[part][region][list_y[0]]
                variable_list_dic[part][region][list_y[0]] = append_new_vars(
                    df_base, df_other)
                                       
# Append variable dfs over regions 
for part in survey_parts:
    for region in ['cabecera', 'resto']:
        df_other = variable_list_dic[part][region]['15']
        df_base = variable_list_dic[part]['area']['15']
        variable_list_dic[part]['area']['15'] = append_new_vars(
            df_base, df_other)
                        
# Reindex dfs with list of numbers in ascending order
for part in survey_parts:
    new_index = list(range(0, len(variable_list_dic[part]['area']['15'])))
    variable_list_dic[part]['area']['15'].reindex(new_index)
    
# Create new dictionary with variable info with only the concatenated dfs over 
# years and regions 
var_info= {part:variable_list_dic[part]['area']['15'] for part in survey_parts}

# Manually fill in missing entries for the variables added 

# vivienda_y_hogares
var_info['vivienda_y_hogares'].loc[var_info['vivienda_y_hogares']['name']=='clase',
                             ['new_name','new_dtype']] = 'clase', 'category'
var_info['vivienda_y_hogares'].loc[
                     var_info['vivienda_y_hogares']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'

# otros_ingresos
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661', 
              ['new_name','new_dtype']] = 'mon_help_from_institutions', 'float'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s1',
     ['new_name','new_dtype']] = 'masfamiliasenaccion_last_12_m', 'category'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s1a1',
    ['new_name','new_dtype']] = 'masfamiliasenaccion_last_12_m_amount', 'float'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s2',
         ['new_name','new_dtype']] = 'jovenesenaccion_last_12_m', 'category'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s2a1',
        ['new_name','new_dtype']] = 'jovenesenaccion_last_12_m_amount', 'float'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s3',
           ['new_name','new_dtype']] = 'colombiamayor_last_12_m', 'category'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s3a1',
          ['new_name','new_dtype']] = 'colombiamayor_last_12_m_amount', 'float'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s4',
                   ['new_name','new_dtype']] = 'other_last_12_m', 'category'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s4a1',
                  ['new_name','new_dtype']] = 'other_last_12_m_name', 'str'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='p1661s4a2',
                  ['new_name','new_dtype']] = 'other_last_12_m_amount', 'float'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='clase',
                             ['new_name','new_dtype']] = 'clase', 'category'
var_info['otros_ingresos'].loc[var_info['otros_ingresos']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'

# otras_actividades_y_ayudas_en_la_semana
var_info['otras_actividades_y_ayudas_en_la_semana'].loc[
          var_info['otras_actividades_y_ayudas_en_la_semana']['name']=='clase',
          ['new_name','new_dtype']] = 'clase', 'category'
var_info['otras_actividades_y_ayudas_en_la_semana'].loc[
    var_info['otras_actividades_y_ayudas_en_la_semana']['name']=='area_department', 
    ['new_name','new_dtype']] = 'area_department', 'category'

# ocupados
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1800',
                     ['new_name','new_dtype']] = 'has_employees', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1800s1',
                    ['new_name','new_dtype']] = 'has_employees_amount', 'Int64'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1801s1',
                       ['new_name','new_dtype']] = 'workers_w_payment', 'Int64'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1801s2',
                         ['new_name','new_dtype']] = 'partners', 'Int64'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1801s3',
                 ['new_name','new_dtype']] = 'workers_without_payment', 'Int64'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1802',
       ['new_name','new_dtype']] = 'offers_services_or_products', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1805',
 ['new_name','new_dtype']] = 'changes_from_freelancer_to_employee', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1879',
              ['new_name','new_dtype']] = 'main_reason_freelancing', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1880',
        ['new_name','new_dtype']] = 'main_reason_quitting_last_job', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1881',
          ['new_name','new_dtype']] = 'main_commuting_transport', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='p1882',
                 ['new_name','new_dtype']] = 'regular_communting_time', 'Int64'
var_info['ocupados'].loc[var_info['ocupados']['name']=='clase',
                             ['new_name','new_dtype']] = 'clase', 'category'
var_info['ocupados'].loc[var_info['ocupados']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'

# inactivos
var_info['inactivos'].loc[var_info['inactivos']['name']=='p744',
 ['new_name','new_dtype']] = 'available_last_week_to_start_working', 'category'
var_info['inactivos'].loc[var_info['inactivos']['name']=='p1807',
               ['new_name','new_dtype']] = 'min_monthly_wage_expected', 'float'
var_info['inactivos'].loc[var_info['inactivos']['name']=='p1884',
         ['new_name','new_dtype']] = 'weekly_hours_available_for_work', 'Int64'
var_info['inactivos'].loc[var_info['inactivos']['name']=='p6921',
          ['new_name','new_dtype']] = 'contributes_to_pension_fund', 'category'
var_info['inactivos'].loc[var_info['inactivos']['name']=='clase',
                          ['new_name','new_dtype']] = 'clase', 'category'
var_info['inactivos'].loc[var_info['inactivos']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'
var_info['inactivos'].loc[var_info['inactivos']['name']=='area_resto',
                        ['new_name','new_dtype']] = 'area_resto', 'category'

# fuerza_de_trabajo
var_info['fuerza_de_trabajo'].loc[var_info['fuerza_de_trabajo']['name']=='raband',
                                  ['new_name','new_dtype']] = 'raband', 'float'
var_info['fuerza_de_trabajo'].loc[var_info['fuerza_de_trabajo']['name']=='clase',
                                  ['new_name','new_dtype']] = 'clase', 'category'
var_info['fuerza_de_trabajo'].loc[var_info['fuerza_de_trabajo']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'

# desocupados
var_info['desocupados'].loc[var_info['desocupados']['name']=='p1519',
          ['new_name','new_dtype']] = 'contributes_to_pension_fund', 'category'
var_info['desocupados'].loc[var_info['desocupados']['name']=='p1806',
               ['new_name','new_dtype']] = 'min_monthly_wage_expected', 'float'
var_info['desocupados'].loc[var_info['desocupados']['name']=='p1883',
                 ['new_name','new_dtype']] = 'current_pension_fund', 'category'
var_info['desocupados'].loc[var_info['desocupados']['name']=='clase',
                            ['new_name','new_dtype']] = 'clase', 'category'
var_info['desocupados'].loc[var_info['desocupados']['name']=='area_department',
                   ['new_name','new_dtype']] = 'area_department', 'category'

# caracteristicas_generales_personas
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6269',
    ['new_name','new_dtype']] = 'graduated_from_esc_normal_superior', 'category'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6071',
    ['new_name','new_dtype']] = 'spouse_resides_at_same_house', 'category'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6071s1',
    ['new_name','new_dtype']] = 'orden_spouse', 'Int64'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6081',
    ['new_name','new_dtype']] = 'father_resides_at_same_house', 'category'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6081s1',
    ['new_name','new_dtype']] = 'orden_father', 'Int64'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6083',
    ['new_name','new_dtype']] = 'mother_resides_at_same_house', 'category'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='p6083s1',
    ['new_name','new_dtype']] = 'orden_mother', 'Int64'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='clase',
    ['new_name','new_dtype']] = 'clase', 'category'
var_info['caracteristicas_generales_personas'].loc[
    var_info['caracteristicas_generales_personas']['name']=='area_department',
    ['new_name','new_dtype']] = 'area_department', 'category'

# Change all string numbers to int or float 
dict_to_num = {}
for part in survey_parts:
    dict_to_num[part] = var_info[part].loc[
        var_info[part]['new_dtype']!='str']['name'].tolist()
    
for part in survey_parts:
    for column in dict_to_num[part]:
        geih_parts[part][column] = pd.to_numeric(
        geih_parts[part][column], errors='coerce')
        
# Save dfs as .pickle file
Path.mkdir((Path(__file__)/'../build/Colombia/output_dfs').resolve())
for key in geih_parts.keys():
    path_1 = '../build/Colombia/output_dfs/' + key + '.pickle'
    path_2 = '../build/Colombia/output_dfs/var_info_' + key + '.pickle'
    geih_parts[key].to_pickle((Path(__file__)/path_1).resolve())
    var_info[key].to_pickle((Path(__file__)/path_2).resolve())    
  

