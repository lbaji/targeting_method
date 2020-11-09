# -*- coding: utf-8 -*-
"""This module creates a smaller dataframe with only the data needed for the 
regressions.

"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from os import listdir
import json

__file__ = 'create_variables.py'

# COLOMBIA

# Load dfs 
survey_parts = []
for f in listdir((Path(__file__)/'../info/Colombia/geih_variables_list').resolve()):
    if f.startswith('geih_15_area'):
        survey_parts.append(f[13:-4])

geih_parts = {}
for part in survey_parts:
    path = '../build/Colombia/output_dfs/' + part + '_final.pickle'
    pickle_in = open((Path(__file__)/path).resolve(), 'rb')
    geih_parts[part] = pickle.load(pickle_in)
    
# Load subset of columns to filter the dataframes 
with open((Path(__file__)/'../info/Colombia/merge_vars.json').resolve(), 'r') as f:
    merge_vars = json.load(f)

dict_to_concat = {key:geih_parts[key][merge_vars[key]] for key 
                  in list(merge_vars.keys())}

# Concatenate only needed variables
a = pd.merge(dict_to_concat['caracteristicas_generales_personas'], 
         dict_to_concat['desocupados'], how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

b = pd.merge(dict_to_concat['ocupados'], 
         dict_to_concat['fuerza_de_trabajo'], how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

c = pd.merge(dict_to_concat['otras_actividades_y_ayudas_en_la_semana'], 
         dict_to_concat['otros_ingresos'], how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

d = pd.merge(a, b, how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

e = pd.merge(c, d, how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

f = pd.merge(e, dict_to_concat['inactivos'], how='outer',
         on=['month_letters', 'year', 'directorio', 
        'secuencia_p', 'orden', 'household', 'region', 
        'clase', 'area_city', 'area_department'])

geih = pd.merge(f, dict_to_concat['vivienda_y_hogares'],
        how='outer', on=['month_letters', 'year', 
        'directorio', 'secuencia_p', 'household', 
        'region', 'clase', 'area_city', 'area_department'])  

# Create new columns containing info from same variable from different dfs and 
# drop the extra columns
geih.loc[geih['after_last_job_searched_for_job_or_own_business_x'].notna(), 
         'after_last_job_searched_for_job_or_own_business'] = geih[
             'after_last_job_searched_for_job_or_own_business_x']

geih.loc[geih['after_last_job_searched_for_job_or_own_business_y'].notna(), 
         'after_last_job_searched_for_job_or_own_business'] = geih[
             'after_last_job_searched_for_job_or_own_business_y']
             
geih.loc[geih['pensions_fund_type_x'].notna(), 
         'pensions_fund_type'] = geih['pensions_fund_type_x']

geih.loc[geih['pensions_fund_type_y'].notna(), 
         'pensions_fund_type'] = geih['pensions_fund_type_y']

geih.drop(columns=['pensions_fund_type_x', 'pensions_fund_type_y', 
                   'after_last_job_searched_for_job_or_own_business_x',
                   'after_last_job_searched_for_job_or_own_business_y'], 
         inplace=True)

# Create hid and pid 
geih_ids = geih[['directorio', 'household', 'orden']].astype('str')

geih['hid'] = geih_ids['directorio'] + geih_ids['household']
geih['pid'] = geih_ids['directorio'] + geih_ids['household'] + geih_ids['orden']

geih = geih.astype({'hid': 'int64',
                    'pid': 'int64'})

# OCUPADOS
# Calculate total monthly income, i.e. all types of wage income plus any other 
# type of income (pensions, subsidies)
geih['monthly_total_income'] = geih[['gross_wage_last_month', 
    'monthly_wage_secondary_job', 'food_as_part_of_wage_last_month_est_amount', 
    'dwelling_as_part_of_wage_last_month_est_amount', 
    'employers_transport_to_go_to_work_est_amount', 
    'goods_or_benefits_as_wage_last_month_est_amount']].sum(axis=1, min_count=1)

geih['monthly_total_income_subsidies'] = geih['monthly_total_income']

# Monthly (subsidy) variables
dict_monthly = {
    'extra_hours': ['paid_extra_hours_last_month_amount', 
                    'paid_extra_hours_included_in_gross_wage'],
     'transport': ['subsidy_for_transport_amount', 
                   'subsidy_for_transport_included_in_gross_wage'],
     'children': ['subsidy_for_children_amount', 
                  'subsidy_for_children_included_in_gross_wage'],
     'education': ['subsidy_for_education_amount', 
                   'subsidy_for_education_included_in_gross_wage'],
     'food': ['subsidy_for_food_amount', 
              'subsidy_for_food_included_in_gross_wage'],
     'bonus_m': ['monthly_bonus_last_month_amount', 
                 'monthly_bonus_included_in_gross_wage'],
     'boni': ['boni_last_month_amount', 'boni_included_in_gross_wage']}

for key, value in dict_monthly.items():
    geih.loc[(geih[value[1]]=='no'), key] = geih[value[0]]

# Yearly total income (monthly_total_income without rest of variables added is needed)
geih['yearly_total_income'] = geih[['monthly_total_income', 
                                    'extra_hours', 'transport']].sum(axis=1, min_count=1)

geih['yearly_total_income_subsidies'] = geih[['monthly_total_income_subsidies', 
                                    'extra_hours', 'transport']].sum(axis=1, min_count=1)

# Yearly bonus variables converted into monthly amounts
dict_yearly = {'service': 'service_bonus_last_12_months_amount', 
               'travel': 'travel_payments_or_boni_last_12_months_amount',
               'leave': 'leave_bonus_last_12_months_amount',
               'christmas': 'christmas_bonus_last_12_months_amount',
               'temp_profits': 'net_profits_last_12_months'}

for key, value in dict_yearly.items():
    geih[key] = geih[value]/12
    geih[key] = geih[key].round()

# 'net_profits_last_month' & 'net_profits_last_12_months'

# Correct 'net_profits_last_month' with 'payment_corresponds_to_months', 
# for some obs the latter equals up to 12 months
geih['net_profits_last_month'] = geih['net_profits_last_month'].divide(
                                 geih['payment_corresponds_to_months'])

# Fill nan in the column 'net_profits_last_month' with 1/12 of 'net_profits_last_12_months' 
geih['net_profits_last_month'] = geih['net_profits_last_month'].fillna(
    geih['temp_profits']) 
    
# Add to 'monthly_total_income' all auxiliary variables 
geih['monthly_total_income'] = geih[['monthly_total_income', 'extra_hours',
    'transport', 'bonus_m', 'boni', 'service', 'travel', 'leave', 'christmas', 
    'net_profits_last_month']].sum(axis=1, min_count=1)

geih['monthly_total_income_subsidies'] = geih[['monthly_total_income_subsidies', 
    'extra_hours', 'transport', 'children', 'education', 'food', 'bonus_m', 
    'boni', 'service', 'travel', 'leave', 'christmas', 
    'net_profits_last_month']].sum(axis=1, min_count=1)

geih['monthly_total_income'] = geih['monthly_total_income'].round()
geih['monthly_total_income_subsidies'] = geih['monthly_total_income_subsidies'].round()

# Calculate total yearly income, i.e. all types of wage income plus 
# any other type of income (pensions, subsidies)
geih['yearly_total_income'] = geih['yearly_total_income'].mul(
    geih['months_worked_in_last_12_months'])

geih['yearly_total_income_subsidies'] = geih['yearly_total_income_subsidies'].mul(
    geih['months_worked_in_last_12_months'])

geih['yearly_temp'] = geih[['children', 'education', 
                            'food', 'bonus_m']].sum(axis=1, min_count=1)
geih['yearly_temp'] = geih['yearly_temp']*12

geih['yearly_temp_2'] = geih[['bonus_m']]*12

geih['yearly_total_income_subsidies'] = geih[['yearly_total_income_subsidies', 
                                    'yearly_temp', 'boni']].sum(axis=1, min_count=1)

geih['yearly_total_income'] = geih[['yearly_total_income', 'yearly_temp_2', 
                                    'boni']].sum(axis=1, min_count=1)

# Add boni (yearly variables)
geih['yearly_total_income'] = geih[['yearly_total_income', 
                                    'service_bonus_last_12_months_amount', 
                                    'travel_payments_or_boni_last_12_months_amount',
                                    'leave_bonus_last_12_months_amount',
                                    'christmas_bonus_last_12_months_amount'
                                   ]].sum(axis=1, min_count=1)

geih['yearly_total_income_subsidies'] = geih[['yearly_total_income_subsidies', 
                                    'service_bonus_last_12_months_amount', 
                                    'travel_payments_or_boni_last_12_months_amount',
                                    'leave_bonus_last_12_months_amount',
                                    'christmas_bonus_last_12_months_amount'
                                            ]].sum(axis=1, min_count=1)

# 'net_profits_last_month' was already corrected with 'payments_correspond_to_months'!
geih['net_profits_last_12_months'] = geih['net_profits_last_12_months'].fillna(
                                        geih['net_profits_last_month']*12)

# 'payments_for_work_accidents_amount' is yearly
geih['yearly_total_income'] = geih[['yearly_total_income', 'net_profits_last_12_months', 
    'payments_for_work_accidents_amount']].sum(axis=1, min_count=1)
geih['yearly_total_income_subsidies'] = geih[['yearly_total_income_subsidies', 
    'net_profits_last_12_months', 'payments_for_work_accidents_amount']].sum(axis=1, min_count=1)

# Drop auxiliary columns
geih.drop(columns=['extra_hours', 'transport', 'children', 
        'education', 'food', 'bonus_m', 'boni', 'service', 'travel', 
        'leave', 'christmas', 'temp_profits', 'yearly_temp', 
        'yearly_temp_2'], inplace=True)

# OTROS_INGRESOS
# Monthly variables
geih['other_income_monthly'] = geih[['received_alimony_last_month_amount', 
    'received_pension_or_disability_annuity_last_month_amount',
    'received_rent_last_month_amount']].sum(axis=1, min_count=1)

geih['other_income_monthly_subsidies'] = geih['other_income_monthly']

# Yearly variables
dict_yearly = {
    'international': 'received_money_from_international_others_last_12_months_amount',
    'unemployment': 'received_unemployment_money_last_12_months_amount',
    'dividends': 'received_payments_from_interest_or_dividends_last_12_months_amount',
    'institutions': 'received_payments_from_institutions_last_12_months_amount',
    'national': 'received_money_from_national_others_last_12_months_amount',
    'other_money': 'received_money_other_amount',
    'priv_institutions': 'received_payments_from_national_private_institutions_last_12_months_amount',
    'govern_institutions': 'received_payments_from_national_governmental_institutions_last_12_months_amount',
    'intern_institutions': 'received_payments_from_international_institutions_last_12_months_amount',
    'masfamiliasenaccion': 'masfamiliasenaccion_last_12_m_amount',
    'jovenesenaccion': 'jovenesenaccion_last_12_m_amount',
    'colombiamayor': 'colombiamayor_last_12_m_amount',
    'other_help': 'other_last_12_m_amount'}

for key, value in dict_yearly.items():
    geih[key] = geih[value]/12
    geih[key] = geih[key].round()
    
geih['other_income_monthly'] = geih[['other_income_monthly', 'international',
    'unemployment', 'dividends', 'national', 'other_money']].sum(axis=1, min_count=1)
    
geih['other_income_monthly_subsidies'] = geih[['other_income_monthly_subsidies', 
    'international', 'unemployment', 'dividends', 'institutions', 
    'national', 'other_money', 'priv_institutions', 'govern_institutions', 
    'intern_institutions', 'masfamiliasenaccion', 'jovenesenaccion', 
    'colombiamayor', 'other_help']].sum(axis=1, min_count=1)

# Transform to yearly amount
geih['other_income_yearly'] = geih['other_income_monthly']*12
geih['other_income_yearly_subsidies'] = geih['other_income_monthly_subsidies']*12

# Add other_income variables to total income variables
geih['monthly_total_income'] = geih[['monthly_total_income', 
                                    'other_income_monthly']].sum(axis=1, min_count=1)
geih['monthly_total_income_subsidies'] = geih[['monthly_total_income_subsidies', 
                                              'other_income_monthly_subsidies']].sum(axis=1, min_count=1)

geih['yearly_total_income'] = geih[['yearly_total_income', 
                                    'other_income_yearly']].sum(axis=1, min_count=1)
geih['yearly_total_income_subsidies'] = geih[['yearly_total_income_subsidies', 
                                              'other_income_yearly_subsidies']].sum(axis=1, min_count=1)

geih['net_profits_last_12_months'] = geih['net_profits_last_12_months'].fillna(
                                        geih['net_profits_last_month']*12)

geih['yearly_total_income'] = geih['yearly_total_income'].fillna(
                                        geih['monthly_total_income']*12)

# Add column with hh income variables
income_grouped = geih[['hid', 
                       'monthly_total_income',
                       'monthly_total_income_subsidies',
                       'yearly_total_income',
                       'yearly_total_income_subsidies']].groupby('hid').sum()#, 'yearly_total_income']]

geih = geih.join(income_grouped, on=['hid'], rsuffix='_hh')

# Correct for yearly_total_income == np.nan, yearly_total_income_hh == 0.0
# among household heads (yearly_total_income_hh = 0 means all yearly_total_income
# by hid were missings)
geih.loc[(geih['yearly_total_income'].isna())&
         (geih['relationship_to_hh_head']=='household_head')&
         (geih['yearly_total_income_hh']==0.0), 
         'yearly_total_income_hh'] = np.nan

# Create indicator variable for legar waste disposal 
geih.loc[(geih['waste_disposal']=='por recolección pública o privada'), 
        'legal_waste_disposal'] = 'yes'

# Create indicator variable for teenager with child in household
# 1) household head 18- yeas old with child
hid_teen_hh_head = list(geih.loc[(geih['age']<=18.0)&
     (geih['relationship_to_hh_head']=='household_head')]['hid'])

hid_teen_hh_head_w_child = list(geih[geih['hid'].isin(hid_teen_hh_head)].loc[
     geih['relationship_to_hh_head']=='child']['hid'])

# 2) teen child of hh head with child
a = geih.loc[((geih['relationship_to_hh_head']=='child')&
              (geih['age']<18.0)) |#&(geih['age']>11.0))|
             ((geih['relationship_to_hh_head']=='grandchild'))][[
    'hid', 'pid', 'age', 'relationship_to_hh_head']]

grouped_hid = a.groupby('hid')
hid_teen_child_w_child = []
for name, group in grouped_hid:
    if len(group['relationship_to_hh_head'].unique())==2:
        min_c = group.loc[group['relationship_to_hh_head']=='child']['age'].min()
        max_g = group.loc[group['relationship_to_hh_head']=='grandchild']['age'].max()
        diff = min_c - max_g
        if diff > 11.0:
            hid_teen_child_w_child.append(name)

hid_teen_child_u_18_w_child = []
for name, group in geih.loc[(geih['hid'].isin(hid_teen_child_w_child))&
                            (geih['relationship_to_hh_head']=='child')][[
                            'hid', 'age']].groupby('hid'):
    l = len(group['age'])
    comp_array = np.full((l), 18.0)
    comparison = np.less(group['age'], comp_array)
    if np.all(comparison.to_numpy())==True:
        hid_teen_child_u_18_w_child.append(name)

hid_hh_teen_w_child = hid_teen_hh_head_w_child + hid_teen_child_u_18_w_child

geih.loc[(geih['hid'].isin(hid_hh_teen_w_child)), 
         'teen_w_child'] = 'yes'

# Create indicator variable for illiterate adult in hh 
hid_illiterate = list(geih.loc[(geih['age']>=18.0)&(geih['literate']=='no')]['hid'])

geih.loc[(geih['hid']).isin(hid_illiterate), 'illiterate_adult_hh_member'] = 'yes'
#geih[geih['literate']=='no'][['hid', 'age', 'literate', 'illiterate_adult_hh_member']]

# Create indicator variable for overcrowded households
# number_of_rooms & number_of_hh_members
geih.loc[((geih['number_of_bedrooms']==1.0)&(geih['number_of_hh_members']>=3.0))|
         ((geih['number_of_bedrooms']==2.0)&(geih['number_of_hh_members']>=5.0))|
         ((geih['number_of_bedrooms']==3.0)&(geih['number_of_hh_members']>=7.0))|
         ((geih['number_of_bedrooms']==4.0)&(geih['number_of_hh_members']>=9.0))|
         ((geih['number_of_bedrooms']==5.0)&(geih['number_of_hh_members']>=11.0))|
         ((geih['number_of_bedrooms']==6.0)&(geih['number_of_hh_members']>=13.0))|
         ((geih['number_of_bedrooms']==7.0)&(geih['number_of_hh_members']>=15.0))|
         ((geih['number_of_bedrooms']==8.0)&(geih['number_of_hh_members']>=17.0))|
         ((geih['number_of_bedrooms']==9.0)&(geih['number_of_hh_members']>=19.0))|
         ((geih['number_of_bedrooms']==10.0)&(geih['number_of_hh_members']>=21.0))|
         ((geih['number_of_bedrooms']==11.0)&(geih['number_of_hh_members']>=23.0)),
         'overcrowding'] = 'yes'

# Create indicator for indigenous dwelling
geih.loc[(geih['dwelling_type']=='indigenous dwelling'), 
         'indigenous_dwelling'] = 'yes'

# Rename columns
geih.rename(columns={
    'unpaid_work_last_week_raise/grow_animals': 'unpaid_work_last_week_raise_grow_animals',
    'month_uninsured': 'months_uninsured'},
           inplace=True)

# Replace missings with the corresponding opposite value and change 
# dummies encoded with 1 to 'yes' and change highest_degree's & 
# highest_education_tile's encoding to from ordered categorical to numeric
geih.replace({'teen_w_child': {np.nan: 'no'},
              'illiterate_adult_hh_member': {np.nan: 'no'},
              'overcrowding': {np.nan: 'no'},
              'legal_waste_disposal': {np.nan: 'no'},
              'dwelling_type': {'other': np.nan},
              'indigenous_dwelling': {np.nan: 'no'},
              'disease_expenses_coverage': {
                  'empeñaría bienes del hogar': np.nan,         
                  'vendería su vivienda o bienes del hogar': np.nan,
                  'otro': np.nan},
              'old_age_finances_compulsory_pension_payment': {1: 'yes', 
                                                              np.nan: 'no'}, 
              'old_age_finances_voluntary_pension_payment': {1: 'yes', 
                                                              np.nan: 'no'},
              'old_age_finances_savings': {1: 'yes', 
                                           np.nan: 'no'}, 
              'old_age_finances_investments': {1: 'yes', 
                                               np.nan: 'no'}, 
              'old_age_finances_private_pension_insurance': {1: 'yes', 
                                                             np.nan: 'no'},
              'old_age_finances_investments_in_children': {1: 'yes', 
                                                           np.nan: 'no'}, 
              'old_age_finances_nothing': {1: 'yes', 
                                           np.nan: 'no'}, 
              'main_occupation_last_week': {'other': np.nan}, 
              'waste_disposal': {'la eliminan de otra forma': np.nan},
              'highest_degree': {
                  'no_degree': 0,
                  'pre_school': 1,
                  'primary_school': 2,
                  'secondary_school': 3,
                  'high_school': 4,
                  'superior': 5},
              'highest_education_title': {
                  'no_title': 0,
                  'high_school': 1,
                  'technician': 2,
                  'university': 3,
                  'postgraduate': 4}}, inplace=True)

# Replace some values in variable 'disease_expenses_coverage' with missings and
# change highest_degree's & highest_education_tile's encoding to from ordered 
# categorical to numeric
#geih.replace({
#    'disease_expenses_coverage': {
#                  'empeñaría bienes del hogar': np.nan,         
#                  'vendería su vivienda o bienes del hogar': np.nan,
#                  'otro': np.nan},
#    'main_occupation_last_week': {'other': np.nan},
#    'highest_degree': {
#        'no_degree': 0,
#        'pre_school': 1,
#        'primary_school': 2,
#        'secondary_school': 3,
#        'high_school': 4,
#        'superior': 5},
#    'highest_education_title': {
#        'no_title': 0,
#       'high_school': 1,
#        'technician': 2,
#        'university': 3,
#        'postgraduate': 4}}, inplace=True)

# Replace unnecessary missings in variables in the list_vars with 'no'
list_vars = ['food_as_part_of_wage_last_month',
             'dwelling_as_part_of_wage_last_month',
             'employers_transport_to_go_to_work',
             'goods_or_benefits_as_wage_last_month']

for var in list_vars:
    geih.loc[((geih[var].isna())&
             (geih[var + '_est_amount'].isna())&
             (geih['main_occupation_last_week']=='working')), var] = 'no'
    
# Replace missings in 'highest_education_title' with information from 
# 'highest_degree'
    
geih.loc[(geih['highest_education_title'].isna())&
         (geih['highest_degree']<=3), 'highest_education_title'] = 0.0

geih.loc[(geih['highest_education_title'].isna())&
         (geih['highest_degree']==4), 'highest_education_title'] = 1.0

# Replace missings with info from other variables
# question asked to everyone +10
var_list = ['received_rent_last_month', 
            'received_pension_or_disability_annuity_last_month', 
            'received_payments_from_interest_or_dividends_last_12_months', 
            'received_unemployment_money_last_12_months',
            'received_money_from_national_others_last_12_months',
            'received_money_from_international_others_last_12_months']
for var in var_list:
    geih.loc[(geih[var].isna())&
         (geih[var + '_amount'].isna()),
         var] = 'no'

    geih.loc[(geih[var].isna())&
         (geih[var+ '_amount'].notna()&
         (geih[var + '_amount']>0.0)),
         var] = 'yes'

# Create new variables yearly_total_income_pc (per capita) and 
# ln_yearly_total_income_pc (natural logarith of the per capita variable)
geih['yearly_total_income_pc'] = geih['yearly_total_income_hh'].div(
    geih['number_of_hh_members'])
geih['ln_yearly_total_income_pc'] = np.log(geih['yearly_total_income_pc'])
geih.loc[
    geih['ln_yearly_total_income_pc']<0.0, 'ln_yearly_total_income_pc'] = 0.0

# Save df as .pickle file
path = '../build/Colombia/output_dfs/reg_data.pickle'
geih.to_pickle((Path(__file__)/path).resolve()) 