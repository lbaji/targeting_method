
"""This module  runs all three functions (ols, quantile, random forest) from 
module "index_replication.functions.py", and also produces tables with the 
metrics of general model fit and targeting power.
 

"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from index_replication.functions import rforest_reg, ols_reg, quantile_reg #plot_predictions, plot_reg_errors, 

__file__ = 'index_replication.py'

# Load regressions' data
path = '../build/Colombia/output_dfs/reg_data.pickle'
pickle_in = open((Path(__file__)/path).resolve(), 'rb')
geih = pickle.load(pickle_in)

years = ['2015', '2016', '2017', '2018', '2019']
data_dict = {y:geih[(geih['relationship_to_hh_head']=='household_head')&
            (geih['year']==int(y))][['ln_yearly_total_income_pc', 'age', 'sex', 
                'relationship_status', 'passed_year_in_highest_degree', 'literate',
                'highest_education_title', 'highest_degree', 'number_of_rooms', 
                'number_of_hh_members', 'overcrowding',
                'main_occupation_last_week', 
                'health_insurance_type', 
                'teen_w_child', 'legal_waste_disposal', 'indigenous_dwelling',
                'hh_has_vacuum_or_polisher', 'hh_has_air_conditioning',
                'hh_has_cable_tv', 'hh_has_bicycle', 'hh_has_motorcycle',
                'hh_has_car', 'hh_has_holiday_home', 'hh_has_internet', 
                'hh_has_washing_machine', 'hh_has_stove', 'hh_has_oven',
                'hh_has_microwave', 'food_as_part_of_wage_last_month',
                'dwelling_as_part_of_wage_last_month', 
                'employers_transport_to_go_to_work',
                'goods_or_benefits_as_wage_last_month',
                'disease_expenses_coverage',
                'unpaid_work_last_week_elderly_or_handicap_care', 
                'months_working_at_current_employment',
                'usual_weekly_hours_worked', 
                'department', 
                'received_payments_from_interest_or_dividends_last_12_months', 
                'stable_job',
                'joined_any_trade_union', 
                'satisfied_w_social_security_benefits',
                'months_worked_in_last_12_months', 
                'had_other_job_before_current_one', 
                'secondary_jobs_last_week',
                'willing_to_change_job',
                'compatible_working_schedule_and_family_responsibilities', 
                'unpaid_work_last_week_raise_grow_animals',
                'unpaid_work_last_week_own_housework',
                'unpaid_work_last_week_childcare', 
                'unpaid_work_last_week_training_courses'  
                ]].dropna() for y in years}

data_dict_by_reg = {c:{y:geih[(geih['relationship_to_hh_head']=='household_head')&
                (geih['year']==int(y))&(geih['clase']==c)][[
                'ln_yearly_total_income_pc', 'age', 'sex', 
                'relationship_status', 'passed_year_in_highest_degree', 'literate',
                'highest_education_title', 'highest_degree', 'number_of_rooms', 
                'number_of_hh_members', 'overcrowding',
                'main_occupation_last_week', 
                'health_insurance_type', 
                'teen_w_child', 'legal_waste_disposal', 'indigenous_dwelling',
                'hh_has_vacuum_or_polisher', 'hh_has_air_conditioning',
                'hh_has_cable_tv', 'hh_has_bicycle', 'hh_has_motorcycle',
                'hh_has_car', 'hh_has_holiday_home', 'hh_has_internet', 
                'hh_has_washing_machine', 'hh_has_stove', 'hh_has_oven',
                'hh_has_microwave', 'food_as_part_of_wage_last_month',
                'dwelling_as_part_of_wage_last_month', 
                'employers_transport_to_go_to_work',
                'goods_or_benefits_as_wage_last_month',
                'disease_expenses_coverage',
                'unpaid_work_last_week_elderly_or_handicap_care', 
                'months_working_at_current_employment',
                'usual_weekly_hours_worked', 
                'department', 
                'received_payments_from_interest_or_dividends_last_12_months', 
                'stable_job',
                'joined_any_trade_union', 
                'satisfied_w_social_security_benefits',
                'months_worked_in_last_12_months', 
                'had_other_job_before_current_one', 
                'secondary_jobs_last_week',
                'willing_to_change_job',
                'compatible_working_schedule_and_family_responsibilities', 
                'unpaid_work_last_week_raise_grow_animals',
                'unpaid_work_last_week_own_housework',
                'unpaid_work_last_week_childcare', 
                'unpaid_work_last_week_training_courses'  
                ]].dropna() for y in years} for c in ['urban', 'rural']}

# Reindex dfs with list of numbers in ascending order
for year in years:
    new_index = list(range(0, len(data_dict[year])))
    data_dict[year].index = new_index
    for c in ['urban', 'rural']:
        new_index_reg = list(range(0, len(data_dict_by_reg[c][year])))
        data_dict_by_reg[c][year].index = new_index_reg
        
# RANDOM FOREST
# All obs
rf_dict_predictions_ts_all, rf_mean_squared_errors_ts_all, rf_r_squareds_ts_all, rf_mean_abs_errors_ts_all, dict_y_test_all, dict_X_test_all, dict_y_train_all, dict_X_train_all = rforest_reg(years, data_dict)

# Urban obs
rf_dict_predictions_ts_urban, rf_mean_squared_errors_ts_urban, rf_r_squareds_ts_urban, rf_mean_abs_errors_ts_urban, dict_y_test_urban, dict_X_test_urban, dict_y_train_urban, dict_X_train_urban = rforest_reg(years, data_dict_by_reg['urban']) 

# Rural obs
rf_dict_predictions_ts_rural, rf_mean_squared_errors_ts_rural, rf_r_squareds_ts_rural, rf_mean_abs_errors_ts_rural, dict_y_test_rural, dict_X_test_rural, dict_y_train_rural, dict_X_train_rural = rforest_reg(years, data_dict_by_reg['rural'])

dict_y_test = {'all': dict_y_test_all,
               'urban': dict_y_test_urban,
               'rural': dict_y_test_rural}

dict_X_test = {'all': dict_X_test_all,
               'urban': dict_X_test_urban,
               'rural': dict_X_test_rural}

dict_y_train = {'all': dict_y_train_all,
               'urban': dict_y_train_urban,
               'rural': dict_y_train_rural}

dict_X_train = {'all': dict_X_train_all,
               'urban': dict_X_train_urban,
               'rural': dict_X_train_rural}

# Create 2 dictionary of dictionaries. The first one is the income dictionary. It contains sub-dictionaries
# with subsamples of the data [all_obs, urban_obs, rural_obs]. Each subdictionary contains the income 
# observations for every year

data_dict_test = {}
data_dict_train = {}
dict_income_test = {}
dict_income_train = {}
subsamples = ['all', 'urban', 'rural']
for s in subsamples:
    dict_income_test[s] = {}
    dict_income_train[s] = {}
    data_dict_test[s] = {}
    data_dict_train[s] = {}
    for year in years:
        dict_income_test[s][year] = dict_y_test[s][year].to_frame(name='ln_yearly_total_income_pc')
        dict_income_train[s][year] = dict_y_train[s][year].to_frame(name='ln_yearly_total_income_pc')
        dict_income_test[s][year].index = list(range(0, len(dict_income_test[s][year])))
        dict_income_train[s][year].index = list(range(0, len(dict_income_train[s][year])))
        dict_X_test[s][year].index = list(range(0, len(dict_X_test[s][year])))
        dict_X_train[s][year].index = list(range(0, len(dict_X_train[s][year])))
        data_dict_test[s][year] = pd.concat([dict_income_test[s][year], dict_X_test[s][year]], axis=1)
        data_dict_train[s][year] = pd.concat([dict_income_train[s][year], dict_X_train[s][year]], axis=1)
        
for s in subsamples:
    for year in years:
        data_dict_test[s][year].rename(columns={
           'department[T.Atlántico]': 'department_Atlántico', 'department[T.Bogotá, d.C.]': 'department_Bogotá', 
           'department[T.Bolívar]': 'department_Bolívar', 'department[T.Boyacá]': 'department_Boyacá', 
           'department[T.Caldas]': 'department_Caldas', 'department[T.Caquetá]': 'department_Caquetá',
           'department[T.Cauca]': 'department_Cauca', 'department[T.Cesar]': 'department_Cesar', 
           'department[T.Chocó]': 'department_Chocó', 'department[T.Cundinamarca]': 'department_Cundinamarca', 
           'department[T.Córdoba]': 'department_Córdoba', 'department[T.Huila]': 'department_Huila', 
           'department[T.La guajira]': 'department_Guajira', 'department[T.Magdalena]': 'department_Magdalena', 
           'department[T.Meta]': 'department_Meta', 'department[T.Nariño]': 'department_Nariño',
           'department[T.Norte de santander]': 'department_Norte_de_Santander', 
           'department[T.Quindio]': 'department_Quindio', 'department[T.Risaralda]': 'department_Risaralda', 
           'department[T.Santander]': 'department_Santander', 'department[T.Sucre]': 'department_Sucre', 
           'department[T.Tolima]': 'department_Tolima', 'department[T.Valle del cauca]': 'department_Valle_del_Cauca',  
           'main_occupation_last_week[T.in education]': 'main_occupation_last_week_in_education',
           'main_occupation_last_week[T.searching for work]': 'main_occupation_last_week_searching_for_work',
           'main_occupation_last_week[T.working]': 'main_occupation_last_week_working', 
           'disease_expenses_coverage[T.con ayudas de los hijos o familiares]': 'disease_expenses_coverage_hijos_o_familiares',
           'disease_expenses_coverage[T.con otro tipo de seguro o cubrimiento]': 'disease_expenses_coverage_seguro',
           'disease_expenses_coverage[T.es afiliado a un régimen subsidiado de salud]': 'disease_expenses_coverage_subsidized',
           'disease_expenses_coverage[T.es afiliado como cotizante a un régimen contributivo de salud (EPS)]': 'disease_expenses_coverage_eps',
           'disease_expenses_coverage[T.es beneficiario de un afiliado]': 'disease_expenses_coverage_beneficiary',
           'disease_expenses_coverage[T.no lo ha considerado]': 'disease_expenses_coverage_not_considered_yet',
           'disease_expenses_coverage[T.no tiene recursos]': 'disease_expenses_coverage_no_resources',
           'disease_expenses_coverage[T.pidiendo dinero prestado]': 'disease_expenses_coverage_borrowing_money', 
           'sex[T.male]': 'male', 'overcrowding[T.yes]': 'overcrowding',
           'health_insurance_type[T.special]': 'health_insurance_type_special',
           'health_insurance_type[T.subsidized]': 'health_insurance_type_subsidized',
           'unpaid_work_last_week_elderly_or_handicap_care[T.yes]': 'unpaid_work_last_week_elderly_or_handicap_care',
           'relationship_status[T.married]': 'relationship_status_married',
           'relationship_status[T.not_married_and_less_than_two_year_relationship]': 'relationship_status_not_married_and_less_than_two_year_relationship',
           'relationship_status[T.not_married_and_more_than_two_year_relationship]': 'relationship_status_not_married_and_more_than_two_year_relationship',
           'relationship_status[T.single]': 'relationship_status_single', 
           'relationship_status[T.widowed]': 'relationship_status_widowed',
           'legal_waste_disposal[T.yes]': 'legal_waste_disposal', 
           'hh_has_vacuum_or_polisher[T.yes]': 'hh_has_vacuum_or_polisher',
           'hh_has_air_conditioning[T.yes]': 'hh_has_air_conditioning', 
           'hh_has_cable_tv[T.yes]': 'hh_has_cable_tv',
           'hh_has_bicycle[T.yes]': 'hh_has_bicycle', 
           'hh_has_motorcycle[T.yes]': 'hh_has_motorcycle',
           'hh_has_car[T.yes]': 'hh_has_car', 
           'hh_has_holiday_home[T.yes]': 'hh_has_holiday_home',
           'hh_has_internet[T.yes]': 'hh_has_internet', 
           'hh_has_washing_machine[T.yes]':  'hh_has_washing_machine',
           'hh_has_stove[T.yes]': 'hh_has_stove', 
           'hh_has_oven[T.yes]': 'hh_has_oven', 
           'hh_has_microwave[T.yes]': 'hh_has_microwave',
           'food_as_part_of_wage_last_month[T.yes]': 'food_as_part_of_wage_last_month',
           'dwelling_as_part_of_wage_last_month[T.yes]': 'dwelling_as_part_of_wage_last_month',
           'employers_transport_to_go_to_work[T.yes]': 'employers_transport_to_go_to_work',
           'goods_or_benefits_as_wage_last_month[T.yes]': 'goods_or_benefits_as_wage_last_month',
           'received_payments_from_interest_or_dividends_last_12_months[T.yes]': 'received_payments_from_interest_or_dividends_last_12_months',
           'stable_job[T.yes]': 'stable_job', 
           'joined_any_trade_union[T.yes]': 'joined_any_trade_union',
           'satisfied_w_social_security_benefits[T.yes]': 'satisfied_w_social_security_benefits',
           'had_other_job_before_current_one[T.yes]': 'had_other_job_before_current_one',
           'secondary_jobs_last_week[T.yes]': 'secondary_jobs_last_week', 
           'willing_to_change_job[T.yes]': 'willing_to_change_job',
           'compatible_working_schedule_and_family_responsibilities[T.yes]': 'compatible_working_schedule_and_family_responsibilities',
           'unpaid_work_last_week_raise_grow_animals[T.yes]': 'unpaid_work_last_week_raise_grow_animals',
           'unpaid_work_last_week_own_housework[T.yes]': 'unpaid_work_last_week_own_housework',
           'unpaid_work_last_week_childcare[T.yes]': 'unpaid_work_last_week_childcare',
           'unpaid_work_last_week_training_courses[T.yes]': 'unpaid_work_last_week_training_courses'}, 
           inplace=True)
        
        data_dict_train[s][year].rename(columns={
           'department[T.Atlántico]': 'department_Atlántico', 'department[T.Bogotá, d.C.]': 'department_Bogotá', 
           'department[T.Bolívar]': 'department_Bolívar', 'department[T.Boyacá]': 'department_Boyacá', 
           'department[T.Caldas]': 'department_Caldas', 'department[T.Caquetá]': 'department_Caquetá',
           'department[T.Cauca]': 'department_Cauca', 'department[T.Cesar]': 'department_Cesar', 
           'department[T.Chocó]': 'department_Chocó', 'department[T.Cundinamarca]': 'department_Cundinamarca', 
           'department[T.Córdoba]': 'department_Córdoba', 'department[T.Huila]': 'department_Huila', 
           'department[T.La guajira]': 'department_Guajira', 'department[T.Magdalena]': 'department_Magdalena', 
           'department[T.Meta]': 'department_Meta', 'department[T.Nariño]': 'department_Nariño',
           'department[T.Norte de santander]': 'department_Norte_de_Santander', 
           'department[T.Quindio]': 'department_Quindio', 'department[T.Risaralda]': 'department_Risaralda', 
           'department[T.Santander]': 'department_Santander', 'department[T.Sucre]': 'department_Sucre', 
           'department[T.Tolima]': 'department_Tolima', 'department[T.Valle del cauca]': 'department_Valle_del_Cauca',  
           'main_occupation_last_week[T.in education]': 'main_occupation_last_week_in_education',
           'main_occupation_last_week[T.searching for work]': 'main_occupation_last_week_searching_for_work',
           'main_occupation_last_week[T.working]': 'main_occupation_last_week_working', 
           'disease_expenses_coverage[T.con ayudas de los hijos o familiares]': 'disease_expenses_coverage_hijos_o_familiares',
           'disease_expenses_coverage[T.con otro tipo de seguro o cubrimiento]': 'disease_expenses_coverage_seguro',
           'disease_expenses_coverage[T.es afiliado a un régimen subsidiado de salud]': 'disease_expenses_coverage_subsidized',
           'disease_expenses_coverage[T.es afiliado como cotizante a un régimen contributivo de salud (EPS)]': 'disease_expenses_coverage_eps',
           'disease_expenses_coverage[T.es beneficiario de un afiliado]': 'disease_expenses_coverage_beneficiary',
           'disease_expenses_coverage[T.no lo ha considerado]': 'disease_expenses_coverage_not_considered_yet',
           'disease_expenses_coverage[T.no tiene recursos]': 'disease_expenses_coverage_no_resources',
           'disease_expenses_coverage[T.pidiendo dinero prestado]': 'disease_expenses_coverage_borrowing_money', 
           'sex[T.male]': 'male', 'overcrowding[T.yes]': 'overcrowding',
           'health_insurance_type[T.special]': 'health_insurance_type_special',
           'health_insurance_type[T.subsidized]': 'health_insurance_type_subsidized',
           'unpaid_work_last_week_elderly_or_handicap_care[T.yes]': 'unpaid_work_last_week_elderly_or_handicap_care',
           'relationship_status[T.married]': 'relationship_status_married',
           'relationship_status[T.not_married_and_less_than_two_year_relationship]': 'relationship_status_not_married_and_less_than_two_year_relationship',
           'relationship_status[T.not_married_and_more_than_two_year_relationship]': 'relationship_status_not_married_and_more_than_two_year_relationship',
           'relationship_status[T.single]': 'relationship_status_single', 
           'relationship_status[T.widowed]': 'relationship_status_widowed',
           'legal_waste_disposal[T.yes]': 'legal_waste_disposal', 
           'hh_has_vacuum_or_polisher[T.yes]': 'hh_has_vacuum_or_polisher',
           'hh_has_air_conditioning[T.yes]': 'hh_has_air_conditioning', 
           'hh_has_cable_tv[T.yes]': 'hh_has_cable_tv',
           'hh_has_bicycle[T.yes]': 'hh_has_bicycle', 
           'hh_has_motorcycle[T.yes]': 'hh_has_motorcycle',
           'hh_has_car[T.yes]': 'hh_has_car', 
           'hh_has_holiday_home[T.yes]': 'hh_has_holiday_home',
           'hh_has_internet[T.yes]': 'hh_has_internet', 
           'hh_has_washing_machine[T.yes]':  'hh_has_washing_machine',
           'hh_has_stove[T.yes]': 'hh_has_stove', 
           'hh_has_oven[T.yes]': 'hh_has_oven', 
           'hh_has_microwave[T.yes]': 'hh_has_microwave',
           'food_as_part_of_wage_last_month[T.yes]': 'food_as_part_of_wage_last_month',
           'dwelling_as_part_of_wage_last_month[T.yes]': 'dwelling_as_part_of_wage_last_month',
           'employers_transport_to_go_to_work[T.yes]': 'employers_transport_to_go_to_work',
           'goods_or_benefits_as_wage_last_month[T.yes]': 'goods_or_benefits_as_wage_last_month',
           'received_payments_from_interest_or_dividends_last_12_months[T.yes]': 'received_payments_from_interest_or_dividends_last_12_months',
           'stable_job[T.yes]': 'stable_job', 
           'joined_any_trade_union[T.yes]': 'joined_any_trade_union',
           'satisfied_w_social_security_benefits[T.yes]': 'satisfied_w_social_security_benefits',
           'had_other_job_before_current_one[T.yes]': 'had_other_job_before_current_one',
           'secondary_jobs_last_week[T.yes]': 'secondary_jobs_last_week', 
           'willing_to_change_job[T.yes]': 'willing_to_change_job',
           'compatible_working_schedule_and_family_responsibilities[T.yes]': 'compatible_working_schedule_and_family_responsibilities',
           'unpaid_work_last_week_raise_grow_animals[T.yes]': 'unpaid_work_last_week_raise_grow_animals',
           'unpaid_work_last_week_own_housework[T.yes]': 'unpaid_work_last_week_own_housework',
           'unpaid_work_last_week_childcare[T.yes]': 'unpaid_work_last_week_childcare',
           'unpaid_work_last_week_training_courses[T.yes]': 'unpaid_work_last_week_training_courses'}, 
           inplace=True)
   
# OLS + QUANTILE REGRESSION
# All obs
quantiles = [0.5]
test_set = 'no'
dict_params_no = {}
ols_dict_params_all = ols_reg(years, data_dict_train['all'], test_set, dict_params_no)
qr_dict_params_all = quantile_reg(years, data_dict_train['all'], quantiles, test_set, dict_params_no)
        
test_set = 'yes'

ols_dict_predictions_ts_all, ols_mean_squared_errors_ts_all, ols_r_squareds_ts_all, ols_mean_absolute_errors_ts_all = ols_reg(years, data_dict_test['all'], test_set, ols_dict_params_all)

q_dict_predictions_ts_all, q_mean_squared_errors_ts_all, q_r_squareds_ts_all, q_mean_absolute_errors_ts_all = quantile_reg(years, data_dict_test['all'], quantiles, test_set, qr_dict_params_all)

# Urban obs 
test_set = 'no'
ols_dict_params_urban = ols_reg(years, data_dict_train['urban'], test_set, dict_params_no)
qr_dict_params_urban = quantile_reg(years, data_dict_train['urban'], quantiles, test_set, dict_params_no)

test_set = 'yes'

ols_dict_predictions_ts_urban, ols_mean_squared_errors_ts_urban, ols_r_squareds_ts_urban, ols_mean_absolute_errors_ts_urban = ols_reg(years, data_dict_test['urban'], test_set, ols_dict_params_urban)

q_dict_predictions_ts_urban, q_mean_squared_errors_ts_urban, q_r_squareds_ts_urban, q_mean_absolute_errors_ts_urban = quantile_reg(years, data_dict_test['urban'], quantiles, test_set, qr_dict_params_urban)

# Rural obs
test_set = 'no'
ols_dict_params_rural = ols_reg(years, data_dict_train['rural'], test_set, dict_params_no)
qr_dict_params_rural = quantile_reg(years, data_dict_train['rural'], quantiles, test_set, dict_params_no)

test_set = 'yes'

ols_dict_predictions_ts_rural, ols_mean_squared_errors_ts_rural, ols_r_squareds_ts_rural, ols_mean_absolute_errors_ts_rural = ols_reg(years, data_dict_test['rural'], test_set, ols_dict_params_rural)

q_dict_predictions_ts_rural, q_mean_squared_errors_ts_rural, q_r_squareds_ts_rural, q_mean_absolute_errors_ts_rural = quantile_reg(years, data_dict_test['rural'], quantiles, test_set, qr_dict_params_rural)

# R-SQUAREDS
# ols
ols_r_squareds_all = ols_r_squareds_ts_all.to_frame(name='all')
ols_r_squareds_urban = ols_r_squareds_ts_urban.to_frame(name='urban')
ols_r_squareds_rural = ols_r_squareds_ts_rural.to_frame(name='rural')

ols_r_squareds = pd.concat([ols_r_squareds_all, ols_r_squareds_urban, ols_r_squareds_rural], axis=1)

# q
q_r_squareds = pd.concat([q_r_squareds_ts_all, q_r_squareds_ts_urban, q_r_squareds_ts_rural], axis=1,
                         keys=['all', 'urban', 'rural'])

# rf
rf_r_squareds_all = rf_r_squareds_ts_all.to_frame(name='all')
rf_r_squareds_urban = rf_r_squareds_ts_urban.to_frame(name='urban')
rf_r_squareds_rural = rf_r_squareds_ts_rural.to_frame(name='rural')

rf_r_squareds = pd.concat([rf_r_squareds_all, rf_r_squareds_urban, rf_r_squareds_rural], axis=1)

# MEAN ABSOLUTE ERROR
# ols
ols_mean_absolute_errors_all = ols_mean_absolute_errors_ts_all.to_frame(name='all')
ols_mean_absolute_errors_urban = ols_mean_absolute_errors_ts_urban.to_frame(name='urban')
ols_mean_absolute_errors_rural = ols_mean_absolute_errors_ts_rural.to_frame(name='rural')
ols_mae = pd.concat([ols_mean_absolute_errors_all, ols_mean_absolute_errors_urban, 
                       ols_mean_absolute_errors_rural], axis=1)

# q
q_mae = pd.concat([q_mean_absolute_errors_ts_all, q_mean_absolute_errors_ts_urban, 
                   q_mean_absolute_errors_ts_rural], axis=1,
                   keys=['all', 'urban', 'rural'])

# rf
rf_mean_absolute_errors_all = rf_mean_abs_errors_ts_all.to_frame(name='all')
rf_mean_absolute_errors_urban = rf_mean_abs_errors_ts_urban.to_frame(name='urban')
rf_mean_absolute_errors_rural = rf_mean_abs_errors_ts_rural.to_frame(name='rural')
rf_mae = pd.concat([rf_mean_absolute_errors_all, rf_mean_absolute_errors_urban, 
                       rf_mean_absolute_errors_rural], axis=1)

# MEAN SQUARED ERROR
# ols
ols_mean_squared_errors_all = ols_mean_squared_errors_ts_all.to_frame(name='all')
ols_mean_squared_errors_urban = ols_mean_squared_errors_ts_urban.to_frame(name='urban')
ols_mean_squared_errors_rural = ols_mean_squared_errors_ts_rural.to_frame(name='rural')
ols_mse = pd.concat([ols_mean_squared_errors_all, ols_mean_squared_errors_urban, 
                       ols_mean_squared_errors_rural], axis=1)

# q
q_mse = pd.concat([q_mean_squared_errors_ts_all, q_mean_squared_errors_ts_urban, 
                   q_mean_squared_errors_ts_rural], axis=1,
                   keys=['all', 'urban', 'rural'])

# rf
rf_mean_squared_errors_all = rf_mean_squared_errors_ts_all.to_frame(name='all')
rf_mean_squared_errors_urban = rf_mean_squared_errors_ts_urban.to_frame(name='urban')
rf_mean_squared_errors_rural = rf_mean_squared_errors_ts_rural.to_frame(name='rural')
rf_mse = pd.concat([rf_mean_squared_errors_all, rf_mean_squared_errors_urban, 
                    rf_mean_squared_errors_rural], axis=1)

# Create dictionary of dictionaries with all predictions for all subsamples and all years

dict_predictions_ts_all = {
    'ols': ols_dict_predictions_ts_all,
    'qr': q_dict_predictions_ts_all,
    'rf': rf_dict_predictions_ts_all
}
dict_predictions_ts_urban = {
    'ols': ols_dict_predictions_ts_urban,
    'qr': q_dict_predictions_ts_urban,
    'rf': rf_dict_predictions_ts_urban
}
dict_predictions_ts_rural = {
    'ols': ols_dict_predictions_ts_rural,
    'qr': q_dict_predictions_ts_rural,
    'rf': rf_dict_predictions_ts_rural
}

dict_predictions_ts = {
    'all': dict_predictions_ts_all,
    'urban': dict_predictions_ts_urban,
    'rural': dict_predictions_ts_rural
}

# Append income predictions to income dfs for each subsample and each year 
for s in subsamples:
    for year in years:
        dict_income_test[s][year]['ols'] = dict_predictions_ts[s]['ols'][year][0] 
        for i, q in enumerate(quantiles):
            dict_income_test[s][year]['q = ' + str(q)] = dict_predictions_ts[s]['qr'][year][0][i] 
        dict_income_test[s][year]['rf'] = dict_predictions_ts[s]['rf'][year][-1] 
        
#poverty_definitions = ['below_one_third_of_min_wage', 'below_poverty_line']
#ln_corresp_values = [15.182272395306745, 14.781608650773487]

# Create dictionary of dictionaries. Each sub-dictionary contains the given population for each year.
# Each sub-dictionary follows a different definition of poor
total_poor_pop = {}
leakage_ols = {}
leakage_qr = {}
leakage_rf = {}
undercoverage_ols = {}
undercoverage_qr = {}
undercoverage_rf = {}
for s in subsamples:
    total_poor_pop[s] = {}
    leakage_ols[s] = {}
    leakage_qr[s] = {}
    leakage_rf[s] = {}
    undercoverage_ols[s] = {}
    undercoverage_qr[s] = {}
    undercoverage_rf[s] = {}
    for d, value in zip(['below_poverty_line'], [14.781608650773487]): #poverty_definitions, ln_corresp_values):
        total_poor_pop[s][d] = {year:len(dict_income_test[s][year].loc[
            dict_income_test[s][year]['ln_yearly_total_income_pc']<value]) 
                            for year in years}
           
        leakage_ols[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<value)&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                             for year in years}
        
        leakage_rf[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<value)&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                           for year in years}
        
        undercoverage_ols[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=value)&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                                   for year in years}
        
        undercoverage_rf[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=value)&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                           for year in years}
        
        leakage_qr[s][d] = {}
        undercoverage_qr[s][d] = {}
        for q in quantiles:
            leakage_qr[s][d][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]<value)&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)]) 
                                   for year in years}
            
            undercoverage_qr[s][d][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]>=value)&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)]) 
                                   for year in years}
            
# cut-offs version

dict_cutoffs = {}
#for s in subsamples:
#    dict_cutoffs[s] = {}
for year in years:
    dict_cutoffs[year] = {}
    for m in ['ols', 'q = 0.5', 'rf']:
        dict_cutoffs[year][m] = np.quantile(dict_income_test['all'][year][m], q=0.3)
        
# Create dictionary of dictionaries. Each sub-dictionary contains the given population for each year.
# Each sub-dictionary follows a different definition of poor

leakage_ols_co = {}
dif_leakage_ols_co = {}
leakage_qr_co = {}
dif_leakage_qr_co = {}
leakage_rf_co = {}
dif_leakage_rf_co = {}
undercoverage_ols_co = {}
dif_undercoverage_ols_co = {}
undercoverage_qr_co = {}
dif_undercoverage_qr_co = {}
undercoverage_rf_co = {}
dif_undercoverage_rf_co = {}
for s in subsamples:
    leakage_ols_co[s] = {}
    dif_leakage_ols_co[s] = {}
    leakage_qr_co[s] = {}
    dif_leakage_qr_co[s] = {}
    leakage_rf_co[s] = {}
    dif_leakage_rf_co[s] = {}
    undercoverage_ols_co[s] = {}
    dif_undercoverage_ols_co[s] = {}
    undercoverage_qr_co[s] = {}
    dif_undercoverage_qr_co[s] = {}
    undercoverage_rf_co[s] = {}
    dif_undercoverage_rf_co[s] = {}
    #poverty_definitions = ['below_one_third_of_min_wage', 'below_poverty_line']
#ln_corresp_values = [15.182272395306745, 14.781608650773487]
    for d, value in zip(['below_poverty_line'], [14.781608650773487]):#poverty_definitions, ln_corresp_values):
        leakage_ols_co[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                             for year in years}
        
        leakage_ols_co[s]['eligibility_cutoff'] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['ols'])])
                             for year in years}
        
          
        
        dif_leakage_ols_co[s][d] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])-np.exp(value))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                             for year in years}
        
        dif_leakage_ols_co[s]['eligibility_cutoff'] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['ols'])][
            'ln_yearly_total_income_pc'])-np.exp(dict_cutoffs[year]['ols']))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['ols'])][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']<dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['ols'])])
                             for year in years}
        
     #np.sum(np.square((np.exp(14.781608650773487) - np.exp(dict_income_test['all']['2015'].loc[
        #    (dict_income_test['all']['2015']['ols']>=dict_cutoffs['2015']['ols'])&
        #    (dict_income_test['all']['2015']['ln_yearly_total_income_pc']<14.781608650773487)][
        #    'ln_yearly_total_income_pc']))/np.exp(14.781608650773487)))/len(dict_income_test['all']['2015'].loc[
        #    (dict_income_test['all']['2015']['ols']>=dict_cutoffs['2015']['ols'])&
        #    (dict_income_test['all']['2015']['ln_yearly_total_income_pc']<14.781608650773487)])   
    
    
        leakage_rf_co[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                           for year in years}
        
        leakage_rf_co[s]['eligibility_cutoff'] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['rf'])])
                           for year in years}
        
               
        dif_leakage_rf_co[s][d] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])-np.exp(value))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                             for year in years}
                
        dif_leakage_rf_co[s]['eligibility_cutoff'] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['rf'])][
            'ln_yearly_total_income_pc'])-np.exp(dict_cutoffs[year]['rf']))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['rf'])][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']<dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['rf'])])
                             for year in years}
                
        
        undercoverage_ols_co[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                                   for year in years}
        
        undercoverage_ols_co[s]['eligibility_cutoff'] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['ols'])])
                                   for year in years}
        
                
        dif_undercoverage_ols_co[s][d] = {year:np.sum(np.square((np.exp(value) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)][
            'ln_yearly_total_income_pc']))/np.exp(value)))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                             for year in years}
        
        dif_undercoverage_ols_co[s]['eligibility_cutoff'] = {year:np.sum(np.square((np.exp(dict_cutoffs[year]['ols']) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['ols'])][
            'ln_yearly_total_income_pc']))/np.exp(dict_cutoffs[year]['ols'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['ols']>=dict_cutoffs[year]['ols'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['ols'])])
                             for year in years}
                
                
        undercoverage_rf_co[s][d] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                           for year in years}
        
        undercoverage_rf_co[s]['eligibility_cutoff'] = {year:len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['rf'])])
                           for year in years}
              
        dif_undercoverage_rf_co[s][d] = {year:np.sum(np.square((np.exp(value) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)][
            'ln_yearly_total_income_pc']))/np.exp(value)))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                             for year in years}
                
        dif_undercoverage_rf_co[s]['eligibility_cutoff'] = {year:np.sum(np.square((np.exp(dict_cutoffs[year]['rf']) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['rf'])][
            'ln_yearly_total_income_pc']))/np.exp(dict_cutoffs[year]['rf'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['rf']>=dict_cutoffs[year]['rf'])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['rf'])])
                             for year in years}
        
                       
        
        leakage_qr_co[s][d] = {}
        undercoverage_qr_co[s][d] = {}
        dif_leakage_qr_co[s][d] = {}
        dif_undercoverage_qr_co[s][d] = {}
        leakage_qr_co[s]['eligibility_cutoff'] = {}
        undercoverage_qr_co[s]['eligibility_cutoff'] = {}
        dif_leakage_qr_co[s]['eligibility_cutoff'] = {}
        dif_undercoverage_qr_co[s]['eligibility_cutoff'] = {}
        for q in quantiles:

            leakage_qr_co[s][d][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)]) 
                                   for year in years}
            
            leakage_qr_co[s]['eligibility_cutoff'][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['q = ' + str(q)])]) 
                                   for year in years}
            
            
                     
            dif_leakage_qr_co[s][d][q] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])-np.exp(value))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=value)])
                             for year in years}
            
            dif_leakage_qr_co[s]['eligibility_cutoff'][q] = {year:np.sum(np.square((np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['q = ' + str(q)])][
            'ln_yearly_total_income_pc'])-np.exp(dict_cutoffs[year]['q = ' + str(q)]))/np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['q = ' + str(q)])][
            'ln_yearly_total_income_pc'])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]<dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']>=dict_cutoffs[year]['q = ' + str(q)])])
                             for year in years}
                
                
            undercoverage_qr_co[s][d][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)]) 
                                   for year in years}
            
            undercoverage_qr_co[s]['eligibility_cutoff'][q] = {year:len(dict_income_test[s][year].loc[
                (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
                (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)])]) 
                                   for year in years}
            
            
                      
            dif_undercoverage_qr_co[s][d][q] = {year:np.sum(np.square((np.exp(value) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)][
            'ln_yearly_total_income_pc']))/np.exp(value)))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<value)])
                             for year in years}
                
            dif_undercoverage_qr_co[s]['eligibility_cutoff'][q] = {year:np.sum(np.square((np.exp(dict_cutoffs[year]['q = ' + str(q)]) - 
            np.exp(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)])][
            'ln_yearly_total_income_pc']))/np.exp(dict_cutoffs[year]['q = ' + str(q)])))/len(dict_income_test[s][year].loc[
            (dict_income_test[s][year]['q = ' + str(q)]>=dict_cutoffs[year]['q = ' + str(q)])&
            (dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)])])
                             for year in years}
        
# Create df to store leakage and undercoverage rates

rates = []
subsets = []
for rate in ['leakage', 'undercoverage']:
    for s in subsamples:
        rates.append(rate)
        subsets.append(s)

b = [rates, subsets]

methods = ['ols', 'rf']
for q in quantiles:
    methods.append('qr_' + str(q))
    
method = []
y = []
for year in years:
    for m in methods:
        method.append(m)
        y.append(year)

a = [y, method]
multi_index = pd.MultiIndex.from_arrays(a, names=('year', 'method'))

multi_column = pd.MultiIndex.from_arrays(b)

rates = pd.DataFrame(index=multi_index, columns=multi_column)
rates_co = pd.DataFrame(index=multi_index, columns=multi_column)
squared_dif_poverty_line = pd.DataFrame(index=multi_index, columns=multi_column)
squared_dif_co = pd.DataFrame(index=multi_index, columns=multi_column)
rates_co_poverty_line = pd.DataFrame(index=multi_index, columns=multi_column)
rates_co_co = pd.DataFrame(index=multi_index, columns=multi_column)

for year in years: 
    for s in subsamples:
        #poverty_definitions = ['below_one_third_of_min_wage', 'below_poverty_line']
        #ln_corresp_values = [15.182272395306745, 14.781608650773487]
            for d in ['below_poverty_line']:#poverty_definitions:
                #col_l = 'leakage' + '_' + s + '_' + d 
                #col_u = 'undercoverage' + '_' + s + '_' + d 
                
                rates.loc[('ols', year), ('leakage', s)] = leakage_ols[s][d][year]/total_poor_pop[s][d][year]
                rates.loc[('ols', year), ('undercoverage', s)] = undercoverage_ols[s][d][year]/total_poor_pop[s][d][year]
                
                # co
                rates_co.loc[('ols', year), ('leakage', s)] = leakage_ols_co[s][d][year]/total_poor_pop[s][d][year]
                rates_co.loc[('ols', year), ('undercoverage', s)] = undercoverage_ols_co[s][d][year]/total_poor_pop[s][d][year]
                
                rates.loc[('rf', year), ('leakage', s)] = leakage_rf[s][d][year]/total_poor_pop[s][d][year]
                rates.loc[('rf', year), ('undercoverage', s)] = undercoverage_rf[s][d][year]/total_poor_pop[s][d][year]
                
                #co
                rates_co.loc[('rf', year), ('leakage', s)] = leakage_rf_co[s][d][year]/total_poor_pop[s][d][year]
                rates_co.loc[('rf', year), ('undercoverage', s)] = undercoverage_rf_co[s][d][year]/total_poor_pop[s][d][year]
                
                for q in quantiles:
                    rates.loc[('qr_'+str(q), year), ('leakage', s)] = leakage_qr[s][d][q][year]/total_poor_pop[s][d][year]
                    rates.loc[('qr_'+str(q), year), ('undercoverage', s)] = undercoverage_qr[s][d][q][year]/total_poor_pop[s][d][year]
                    
                    # co
                    rates_co.loc[('qr_'+str(q), year), ('leakage', s)] = leakage_qr_co[s][d][q][year]/total_poor_pop[s][d][year]
                    rates_co.loc[('qr_'+str(q), year), ('undercoverage', s)] = undercoverage_qr_co[s][d][q][year]/total_poor_pop[s][d][year]
                    
# 'below_poverty_line'
for year in years: 
    for s in subsamples:
            for d in ['below_poverty_line']:
                #col_l = 'leakage'# + '_' + s + '_' + d 
                #col_u = 'undercoverage'# + '_' + s + '_' + d
                
                rates_co_poverty_line.loc[(year, 'ols'), ('leakage', s)] = leakage_ols_co[s][d][year]/total_poor_pop[s][d][year]
                rates_co_poverty_line.loc[(year, 'ols'), ('undercoverage', s)] = undercoverage_ols_co[s][d][year]/total_poor_pop[s][d][year]
                
                squared_dif_poverty_line.loc[('ols', year), ('leakage', s)] = dif_leakage_ols_co[s][d][year]
                squared_dif_poverty_line.loc[('ols', year), ('undercoverage', s)] = dif_undercoverage_ols_co[s][d][year]
                
                #median_dif_poverty_line.loc[('ols', year), ('leakage', s)] = dif_leakage_ols_co[s][d][year]
                #median_dif_poverty_line.loc[('ols', year), ('undercoverage', s)] = dif_undercoverage_ols_co[s][d][year]
                
                for q in quantiles:
                    rates_co_poverty_line.loc[(year, 'qr q = '+str(q)), ('leakage', s)] = leakage_qr_co[s][d][q][year]/total_poor_pop[s][d][year]
                    rates_co_poverty_line.loc[(year, 'qr q = '+str(q)), ('undercoverage', s)] = undercoverage_qr_co[s][d][q][year]/total_poor_pop[s][d][year]
                    
                    squared_dif_poverty_line.loc[('qr q = '+str(q), year), ('leakage', s)] = dif_leakage_qr_co[s][d][q][year]
                    squared_dif_poverty_line.loc[('qr q = '+str(q), year), ('undercoverage', s)] = dif_undercoverage_qr_co[s][d][q][year]
                    
                    #median_dif_poverty_line.loc[('qr q = '+str(q), year), ('leakage', s)] = dif_leakage_qr_co[s][d][q][year]
                    #median_dif_poverty_line.loc[('qr q = '+str(q), year), ('undercoverage', s)] = dif_undercoverage_qr_co[s][d][q][year]
                
                rates_co_poverty_line.loc[(year, 'rf'), ('leakage', s)] = leakage_rf_co[s][d][year]/total_poor_pop[s][d][year]
                rates_co_poverty_line.loc[(year, 'rf'), ('undercoverage', s)] = undercoverage_rf_co[s][d][year]/total_poor_pop[s][d][year]
                
                squared_dif_poverty_line.loc[('rf', year), ('leakage', s)] = dif_leakage_rf_co[s][d][year]
                squared_dif_poverty_line.loc[('rf', year), ('undercoverage', s)] = dif_undercoverage_rf_co[s][d][year]
                
                #median_dif_poverty_line.loc[('rf', year), ('leakage', s)] = dif_leakage_rf_co[s][d][year]
                #median_dif_poverty_line.loc[('rf', year), ('undercoverage', s)] = dif_undercoverage_rf_co[s][d][year]
                
# poverty definition = eligibility cutoff
for year in years: 
    for s in subsamples:
            for d in ['eligibility_cutoff']:
                #col_l = 'leakage'# + '_' + s + '_' + d 
                #col_u = 'undercoverage'# + '_' + s + '_' + d 
                
                # total_poor_pop[s][d] = {year:len(dict_income_test[s][year].loc[
                # dict_income_test[s][year]['ln_yearly_total_income_pc']<value]) 
                #           for year in years}
                    
                #len(dict_income_test[s][year].loc[
                # dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)]])
                
                rates_co_co.loc[('ols', year), ('leakage', s)] = leakage_ols_co[s][d][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['ols']])
                rates_co_co.loc[('ols', year), ('undercoverage', s)] = undercoverage_ols_co[s][d][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['ols']])
                
                squared_dif_co.loc[('ols', year), ('leakage', s)] = dif_leakage_ols_co[s][d][year]
                squared_dif_co.loc[('ols', year), ('undercoverage', s)] = dif_undercoverage_ols_co[s][d][year]
                
                for q in quantiles:
                    rates_co_co.loc[('qr q = '+str(q), year), ('leakage', s)] = leakage_qr_co[s][d][q][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)]])
                    rates_co_co.loc[('qr q = '+str(q), year), ('undercoverage', s)] = undercoverage_qr_co[s][d][q][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['q = ' + str(q)]])
                    
                    squared_dif_co.loc[('qr q = '+str(q), year), ('leakage', s)] = dif_leakage_qr_co[s][d][q][year]
                    squared_dif_co.loc[('qr q = '+str(q), year), ('undercoverage', s)] = dif_undercoverage_qr_co[s][d][q][year]
                                    
                rates_co_co.loc[('rf', year), ('leakage', s)] = leakage_rf_co[s][d][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['rf']])
                rates_co_co.loc[('rf', year), ('undercoverage', s)] = undercoverage_rf_co[s][d][year]/len(dict_income_test[s][year].loc[
                 dict_income_test[s][year]['ln_yearly_total_income_pc']<dict_cutoffs[year]['rf']])
                
                squared_dif_co.loc[('rf', year), ('leakage', s)] = dif_leakage_rf_co[s][d][year]
                squared_dif_co.loc[('rf', year), ('undercoverage', s)] = dif_undercoverage_rf_co[s][d][year]

df_dict = {'rates': rates, 'rates_co': rates_co, 
           'squared_dif_co': squared_dif_co, 
           'squared_dif_poverty_line': squared_dif_poverty_line, 
           'rates_co_poverty_line': rates_co_poverty_line, 
           'rates_co_co': rates_co_co}

# Save dfs as .pickle file
#Path.mkdir((Path(__file__)/'../build/Colombia/output_dfs').resolve())

for df in ['rates', 'rates_co', 'squared_dif_co', 'squared_dif_poverty_line', 
           'rates_co_poverty_line', 'rates_co_co']:
    path = '../build/Colombia/output_dfs/' + df + '.pickle'
    df_dict[df].to_pickle((Path(__file__)/path).resolve()) 










