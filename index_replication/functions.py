# -*- coding: utf-8 -*-
"""
This module contains the functions needed for the index replication part.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg as qr
import scipy.stats as stats
from patsy import dmatrices
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def plot_predictions(dict_income, dict_income_ml, dict_ml_predictions, 
                     dict_ols_predictions, dict_q_predictions, quantiles, 
                     include_lags, ml_sample):
    """ Plot observed income against its predictions for each year. There is a 
    45° line in each plot to measure prediction power. Return a dictionary with
    years as keys and plots as values. 
    
    Args:
        dict_income: dictionary with years as keys and income series for each 
            year as values 
        dict_income_ml: dictionary with years as keys and income dataframes for
            each year as values, a subsample of dict_income
        dict_ml_predictions: dictionary with years as keys and lists for each 
            year as values. Each list contains the predictions of the year in 
            the corresponding key made with coefficient estimates for the same 
            year and all given lags, i.e. there are n+1 arrays per list 
            with n being the difference between the corresponding year and the 
            base year (2015). For each year, predictions are ordered by the
            models year, in ascending order
        dict_ols_predictions: dictionary with years as keys and lists for each 
            year as values. Each list contains the predictions of the year in 
            the corresponding key made with coefficient estimates for the same 
            year and all the given lags, i.e. there are n+1 arrays per list 
            with n being the difference between the corresponding year and the 
            base year (2015). For each year, predictions are ordered starting 
            from the key and in descending order
        dict_q_predictions: analog to dict_ols_predictions but including an 
            extra dimension for the quantiles, i.e. contained in each list per 
            year there are lists with q arrays. For each year, for each q, 
            predictions are ordered starting from the key and in descending order
        quantiles (list): list of quantiles (floats) used for the predictions
            of the quantile regression
        include_lags (string): if 'no', only the plots with predictions from 
            coefficient estimates of current year will be plotted. If 
            'yes' predictions from coefficient estimates of lagged years
            will also be included in the plots
        ml_sample (string): if 'total' plot ml predictions for the whole sample,
            if 'sub' plot ml predictions for the test subsample only.
            
    """
    origin_pred = ['bethas_current_year', 'bethas_minus1', 'bethas_minus2',
                     'bethas_minus3', 'bethas_minus4']
    years = ['2015', '2016', '2017', '2018', '2019']
    pred_plot_dict = {year:[] for year in years}

    for i, year in enumerate(years): 
        if include_lags == 'no':
            fig = plt.figure(figsize=(10, 10))
        else:
            fig = plt.figure(figsize=(10, 10*(i+1)))

        fig.suptitle('Observed ' + year + ' income against predictions', 
                     fontsize=16)

        if include_lags == 'no':
            ax = fig.add_subplot(1,1,1)
            ax.set_title(origin_pred[0], loc='right')
            ax.set_xlabel('observed income')
            ax.set_ylabel('predictions')

            plt.plot(dict_income[year], dict_income[year], 
                     color='grey')
            
            #if ml_sample=='sub':
            #    ax.scatter(x=dict_income_ml[year], 
            #           y=dict_ml_predictions[year][i],
            #           alpha=0.1, label='Random Forest Regression')
                
            #else:
            #    ax.scatter(x=dict_income[year], 
            #           y=dict_ml_predictions[year][i],
            #           alpha=0.1, label='Random Forest Regression')

            #ax.scatter(x=dict_income[year], 
            #           y=dict_ols_predictions[year][0],
            #           alpha=0.1, label='OLS Regression')

            for q, quantile in enumerate(quantiles):
                ax.scatter(x=dict_income[year],
                           y=dict_q_predictions[year][0][q], alpha=0.1, 
                           label='Quantile Regression, q=' + str(quantile))

            ax.legend()
            pred_plot_dict[year].append(fig)

        else:
            axs = []
            nrows = i + 1
            for j in range(1, nrows + 1): 
                axs.append(fig.add_subplot(nrows, 1, j))
                axs[j-1].set_title(origin_pred[j-1], loc='right')
                axs[j-1].set_xlabel('observed income')
                axs[j-1].set_ylabel('predictions')

                plt.plot(dict_income[year], dict_income[year],
                         color='grey')
                
                if ml_sample=='sub':
                    ax[j-1].scatter(x=dict_income_ml[year], 
                       y=dict_ml_predictions[year][-j],
                       alpha=0.1, label='Random Forest Regression')
                    
                else:
                    ax[j-1].scatter(x=dict_income[year], 
                       y=dict_ml_predictions[year][-j],
                       alpha=0.1, label='Random Forest Regression')

                axs[j-1].scatter(x=dict_income[year], 
                                 y=dict_ols_predictions[year][j-1],
                                 alpha=0.1, label='OLS Regression')
                
                for q, quantile in enumerate(quantiles):
                    axs[j-1].scatter(x=dict_income[year], 
                                     y=dict_q_predictions[year][j-1][q], 
                                     alpha=0.1, 
                                     label='Quantile Regression, q='+str(quantile))
                axs[j-1].legend()
                pred_plot_dict[year].append(fig)
    
    #return pred_plot_dict

def plot_reg_errors(data_dict, indep_var_list, reg, include_lags):
    """ Plot the regression errors of the variable 'ln_yearly_total_income_pc'
    against each independent variable in the indep_var_list. There are n+1 
    plots per year with n being the difference between the corresponding 
    year and the base year (2015). Return a dictionary with independent 
    variables' names as keys and lists with all the plots per ind. variables 
    as values.
    
    Args: 
        data_dict: data dictionary with years as keys and dataframes as values.
            There must be columns with the income errors in each dataframe.
            The column names for these errors must be as in the list 
            "origin_errors", see below
        indep_var_list: list with all independent variables that will be plot 
        against the dependent variable 'ln_yearly_total_income_pc'
        reg (string): equals 'ols' if predictions come from the ols regression
            and 'qr' if predictions come from the quantile regression
        include_lags (string): if 'no', only the plots with prediction error 
            from coefficient estimates of current year will be plotted. If 
            'yes' prediction error from coefficient estimates of lagged years
            will also be included in the plots.
        
    """
    origin_errors = ['error_current_year', 'error_minus1', 'error_minus2',
                     'error_minus3', 'error_minus4']
    years = ['2015', '2016', '2017', '2018', '2019']
    error_plot_dict = {var:[] for var in indep_var_list}
    for var in indep_var_list:
        for i, year in enumerate(years): 
            if include_lags == 'no':
                error_plot_dict[var].append(plt.figure(figsize=(20, 10)))
            else:
                error_plot_dict[var].append(plt.figure(figsize=(20, 10*(i+1))))
            
            error_plot_dict[var][i].suptitle(
                year + ', ' + reg + ' income error against ' + var, fontsize=16)
            
            if include_lags == 'no':
                ax = error_plot_dict[var][i].add_subplot(1,1,1)
                ax.set_title(reg + '_' + origin_errors[0], loc='right')
                ax.set_xlabel(var)
                ax.set_ylabel('income error')
                sns.regplot(x=data_dict[year][var], y=data_dict[year][
                    reg + '_' + origin_errors[0]], ax=ax, order=5, 
                    scatter_kws={'alpha': 0.2}, line_kws={'color':'red'})
            else:
                axs = []
                nrows = i + 1
                for j in range(1, nrows + 1): 
                    axs.append(error_plot_dict[var][i].add_subplot(nrows, 1, j))
                    axs[j-1].set_title(reg + '_' + origin_errors[j-1], loc='right')
                    axs[j-1].set_xlabel(var)
                    axs[j-1].set_ylabel('income error')
                    sns.regplot(x=data_dict[year][var], y=data_dict[year][
                        reg + '_' + origin_errors[j-1]], ax=axs[j-1], order=5, 
                        scatter_kws={'alpha': 0.2}, line_kws={'color':'red'})
                    #axs[j-1].plot(data_dict[year][var], data_dict[year][
                    #    reg + '_' + origin_errors[j-1]], 'o')
    #return error_plot_dict

def rforest_reg(years, data_dict):
    """ Predict yearly total income for every year in years with the 
    corresponding indep. variables using sklearn's RandomForestRegressor.
    Predict values of the dependent variable every year also with all 
    its lagged random forests and calculate r-squareds and mean squared 
    errors to test prediction power of the forest over time. This means that 
    the y's of every year will have n_year - base_year predictions. 
    Return a dataframe with the r-squareds of all predictions, 
    a series with mean squared errors of all predictions, and a dictionary 
    with all predictions and another.
    
    Args:
        years: list with the survey years as strings
        data_dict: dictionary with data separated by survey year.
                The order of the variables is irrelevant as long as
                the dependent variable is the first one
        quantiles: list with the quantiles for the regression  
    
    """    
    arrays = [['2015', '2016', '2016', '2017', '2017', '2017', 
               '2018', '2018', '2018', '2018', '2019', '2019',
               '2019', '2019', '2019'], 
          ['current', 'current', '-1', 'current', '-1', '-2', 
           'current', '-1', '-2', '-3', 'current', '-1', '-2',
           '-3', '-4']]
    multi_index = pd.MultiIndex.from_arrays(arrays, 
                        names=('year', 'origin_coefficients'))
    r_squareds = pd.Series(index=multi_index)
    r_squareds_test_subsample = pd.Series(index=multi_index)
    
    mean_squared_errors = pd.Series(index=multi_index)
    mean_squared_errors_test_subsample = pd.Series(index=multi_index)
    
    mean_abs_errors = pd.Series(index=multi_index)
    mean_abs_errors_test_subsample = pd.Series(index=multi_index)
    
    dict_predictions = {year:[] for year in years}
    dict_predictions_test_subsample = {year:[] for year in years}
    dict_y = {}
    dict_y_test = {}
    dict_y_train = {}
    dict_X = {}
    dict_X_test = {}
    dict_X_train = {}
    for year in years: 
        y, X = dmatrices('ln_yearly_total_income_pc ~ age + sex + \
                     department + passed_year_in_highest_degree + \
                     highest_education_title + highest_degree + number_of_rooms + \
                     number_of_hh_members + overcrowding + \
                     main_occupation_last_week + \
                     health_insurance_type + \
                     disease_expenses_coverage + \
                     unpaid_work_last_week_elderly_or_handicap_care + \
                     relationship_status + \
                     legal_waste_disposal + \
                     hh_has_vacuum_or_polisher + hh_has_air_conditioning + \
                     hh_has_cable_tv + hh_has_bicycle + hh_has_motorcycle + \
                     hh_has_car + hh_has_holiday_home + hh_has_internet + \
                     hh_has_washing_machine + hh_has_stove + hh_has_oven + \
                     hh_has_microwave + food_as_part_of_wage_last_month + \
                     dwelling_as_part_of_wage_last_month + \
                     employers_transport_to_go_to_work + \
                     goods_or_benefits_as_wage_last_month + \
                     months_working_at_current_employment + \
                     usual_weekly_hours_worked + \
                     received_payments_from_interest_or_dividends_last_12_months + \
                     stable_job + \
                     joined_any_trade_union + \
                     satisfied_w_social_security_benefits + \
                     months_worked_in_last_12_months + \
                     had_other_job_before_current_one + \
                     secondary_jobs_last_week + \
                     willing_to_change_job + \
                     compatible_working_schedule_and_family_responsibilities + \
                     unpaid_work_last_week_raise_grow_animals + \
                     unpaid_work_last_week_own_housework + \
                     unpaid_work_last_week_childcare + \
                     unpaid_work_last_week_training_courses',   
                     #np.power(passed_year_in_highest_degree, 2) + \
                     #np.power(highest_degree, 2) + \
                     #np.power(highest_education_title, 2) + \
                     #np.power(highest_degree, 3) + \
                     #np.power(highest_education_title, 3) + \
                     #np.power(number_of_rooms, 2) + \
                     #np.power(number_of_hh_members, 2) + np.power(age, 2) + \
                     #np.power(number_of_bedrooms, 2)', 
    
                     # literate + teen_w_child + indigenous_dwelling +
                     # np.power(passed_year_in_highest_degree, 3) + \
                     data=data_dict[year], return_type='dataframe')
    
        y = pd.Series(y.to_numpy().reshape(y.shape[0],))
        X.drop(columns=['Intercept'], inplace=True)
        dict_y[year] = y
        dict_X[year] = X
                
        X_train, X_test, y_train, y_test = train_test_split(dict_X[year], 
                                dict_y[year], test_size=0.5, random_state=42)
        
        dict_y_test[year] = y_test
        dict_y_train[year] = y_train
        dict_X_test[year] = X_test
        dict_X_train[year] = X_train
         
    for i, year in enumerate(years):    
            
        forest = RandomForestRegressor(n_estimators=500, #criterion='mae',
                                       max_depth=35, min_samples_split=10, 
                                       max_features='log2', random_state=2) 
    
        forest.fit(dict_X_train[year], dict_y_train[year])
                
        dif = int(years[-1]) - int(year)
        for j in range(0, dif+1):
            dict_predictions[str(int(year)+j)].append(
                forest.predict(dict_X[str(int(year)+j)]))
            
            dict_predictions_test_subsample[str(int(year)+j)].append(
                forest.predict(dict_X_test[str(int(year)+j)]))
            
        for j in range(0, dif+1):
            # Calculate r-squareds for predictions with 
            # coefficient estimates from previous year
            res_t = np.sum(np.square(data_dict[str(int(year)+j)][
                'ln_yearly_total_income_pc'] - dict_predictions[str(int(year)+j)][i]))
            tot_t = np.sum(np.square(data_dict[str(int(year)+j)][
                'ln_yearly_total_income_pc'] - data_dict[str(int(year)+j)][
                'ln_yearly_total_income_pc'].mean()))
            r_squareds[str(int(year)+j)][j] = 1 - res_t/tot_t
            
            #only for test sumsample
            res_s = np.sum(np.square(dict_y_test[str(int(year)+j)]
                         - dict_predictions_test_subsample[str(int(year)+j)][i]))
            tot_s = np.sum(np.square(dict_y_test[str(int(year)+j)]- 
                                     dict_y_test[str(int(year)+j)].mean()))
            r_squareds_test_subsample[str(int(year)+j)][j] = 1 - res_s/tot_s
    
            # Calculate the mean squared error
            mean_squared_errors[str(int(year)+j)][j] = np.sum(np.square(
                data_dict[str(int(year)+j)]['ln_yearly_total_income_pc'] - 
                dict_predictions[str(int(year)+j)][i]))/len(data_dict[str(int(year)+j)])
            
            #only for test sumsample
            mean_squared_errors_test_subsample[str(int(year)+j)][j] = np.sum(np.square(
                dict_y_test[str(int(year)+j)] - dict_predictions_test_subsample[
                    str(int(year)+j)][i]))/len(dict_y_test[str(int(year)+j)])
    
            # Calculate the mean absolute error
            mean_abs_errors[str(int(year)+j)][j] = np.sum(np.abs(
                data_dict[str(int(year)+j)]['ln_yearly_total_income_pc'] - 
                dict_predictions[str(int(year)+j)][i]))/len(data_dict[str(int(year)+j)])
            
            #only for test sumsample
            mean_abs_errors_test_subsample[str(int(year)+j)][j] = np.sum(np.abs(
                dict_y_test[str(int(year)+j)] - dict_predictions_test_subsample[
                    str(int(year)+j)][i]))/len(dict_y_test[str(int(year)+j)])
            
    return dict_predictions_test_subsample, mean_squared_errors_test_subsample, \
        r_squareds_test_subsample, mean_abs_errors_test_subsample, \
        dict_y_test, dict_X_test, dict_y_train, dict_X_train 


def quantile_reg(years, data_dict, quantiles, test_set, dict_params):
    """ Regress data from every survey year using statsmodels'
    quantile regression.
    Save coefficient estimates and then predict values of the 
    dependent variable every year with all its lagged coefficient 
    estimates and calculate r-squareds and pseudo r-squareds to 
    test prediction power of estimates over time. This means that 
    the y's of every year will have n_year - base_year predictions. 
    Return a dataframe with the r-squareds and pseudo r-squareds of 
    all predictions, a dataframe with mean squared errors of all predictions, 
    a dictionary with all predictions and another 
    dictionary with the regression summaries for each year.
    
    Args:
        years: list with the survey years as strings
        data_dict: dictionary with data separated by survey year. 
            The order of the variables is irrelevant.
        quantiles: list with the quantiles for the regression  
        test_set: if 'yes' then regress only on test data from random
            forest, if no, on training data
    
    """    
    origin_coef = ['current', '-1', '-2', '-3', '-4']
        
    arrays = [['2015', '2016', '2016', '2017', '2017', '2017', 
                   '2018', '2018', '2018', '2018', '2019', '2019',
                   '2019', '2019', '2019'], 
              ['current', 'current', '-1', 'current', '-1', '-2', 
               'current', '-1', '-2', '-3', 'current', '-1', '-2',
               '-3', '-4']]
    multi_index = pd.MultiIndex.from_arrays(arrays, 
                        names=('year', 'origin_coefficients'))
    columns = []
    for q in quantiles:
        columns.append('q = ' + str(q))
        
    normal_r_squareds = pd.DataFrame(index=multi_index, columns=columns)
    pseudo_r_squareds = pd.DataFrame(index=multi_index, columns=columns)
    mean_squared_errors = pd.DataFrame(index=multi_index, columns=columns)
    mean_abs_errors = pd.DataFrame(index=multi_index, columns=columns)
    
    dict_predictions = {year:[] for year in years}
        
    if test_set=='no': 
        dict_params = {year:[] for year in years}
        for year in years: 
            y, X = dmatrices('ln_yearly_total_income_pc ~ male + \
            department_Atlántico + department_Bogotá + department_Bolívar + \
            department_Boyacá + department_Caldas + department_Caquetá + \
            department_Cauca + department_Cesar + department_Chocó + \
            department_Cundinamarca + department_Córdoba + department_Huila + \
            department_Guajira + department_Magdalena + department_Meta + \
            department_Nariño + department_Norte_de_Santander + \
            department_Quindio + department_Risaralda + department_Santander + \
            department_Sucre + department_Tolima + department_Valle_del_Cauca + \
            main_occupation_last_week_working + \
            disease_expenses_coverage_hijos_o_familiares + \
            disease_expenses_coverage_seguro + \
            disease_expenses_coverage_subsidized + disease_expenses_coverage_eps + \
            disease_expenses_coverage_beneficiary + \
            disease_expenses_coverage_not_considered_yet + \
            disease_expenses_coverage_no_resources + \
            disease_expenses_coverage_borrowing_money + \
            overcrowding + health_insurance_type_special + \
            health_insurance_type_subsidized + \
            unpaid_work_last_week_elderly_or_handicap_care + \
            relationship_status_married + \
            relationship_status_not_married_and_less_than_two_year_relationship + \
            relationship_status_not_married_and_more_than_two_year_relationship + \
            relationship_status_single + relationship_status_widowed + \
            legal_waste_disposal + hh_has_vacuum_or_polisher + \
            hh_has_air_conditioning + hh_has_cable_tv + \
            hh_has_bicycle + hh_has_motorcycle + \
            hh_has_car + hh_has_holiday_home + \
            hh_has_internet + hh_has_washing_machine + \
            hh_has_stove + hh_has_oven + hh_has_microwave + \
            food_as_part_of_wage_last_month + \
            dwelling_as_part_of_wage_last_month + \
            employers_transport_to_go_to_work + \
            goods_or_benefits_as_wage_last_month + \
            received_payments_from_interest_or_dividends_last_12_months + \
            stable_job + joined_any_trade_union + \
            satisfied_w_social_security_benefits + \
            had_other_job_before_current_one + \
            secondary_jobs_last_week + \
            willing_to_change_job + \
            compatible_working_schedule_and_family_responsibilities + \
            unpaid_work_last_week_raise_grow_animals + \
            unpaid_work_last_week_own_housework + \
            unpaid_work_last_week_childcare + \
            unpaid_work_last_week_training_courses + age + \
            passed_year_in_highest_degree + highest_education_title + \
            highest_degree + number_of_rooms + \
            number_of_hh_members + months_working_at_current_employment + \
            usual_weekly_hours_worked + months_worked_in_last_12_months', 
            #main_occupation_last_week_searching_for_work + \
            #main_occupation_last_week_in_education + \
            data=data_dict[year], return_type='dataframe')
                
            mod = qr(y, X)
            for quantile in quantiles:
                result = mod.fit(q=quantile, max_iter=1000)
                dict_params[year].append(result.params)
                
        return dict_params
                        
    else:
        for year in years:
            y, X = dmatrices('ln_yearly_total_income_pc ~ male + \
            department_Atlántico + department_Bogotá + department_Bolívar + \
            department_Boyacá + department_Caldas + department_Caquetá + \
            department_Cauca + department_Cesar + department_Chocó + \
            department_Cundinamarca + department_Córdoba + department_Huila + \
            department_Guajira + department_Magdalena + department_Meta + \
            department_Nariño + department_Norte_de_Santander + \
            department_Quindio + department_Risaralda + department_Santander + \
            department_Sucre + department_Tolima + department_Valle_del_Cauca + \
            main_occupation_last_week_working + \
            disease_expenses_coverage_hijos_o_familiares + \
            disease_expenses_coverage_seguro + \
            disease_expenses_coverage_subsidized + disease_expenses_coverage_eps + \
            disease_expenses_coverage_beneficiary + \
            disease_expenses_coverage_not_considered_yet + \
            disease_expenses_coverage_no_resources + \
            disease_expenses_coverage_borrowing_money + \
            overcrowding + health_insurance_type_special + \
            health_insurance_type_subsidized + \
            unpaid_work_last_week_elderly_or_handicap_care + \
            relationship_status_married + \
            relationship_status_not_married_and_less_than_two_year_relationship + \
            relationship_status_not_married_and_more_than_two_year_relationship + \
            relationship_status_single + relationship_status_widowed + \
            legal_waste_disposal + hh_has_vacuum_or_polisher + \
            hh_has_air_conditioning + hh_has_cable_tv + \
            hh_has_bicycle + hh_has_motorcycle + \
            hh_has_car + hh_has_holiday_home + \
            hh_has_internet + hh_has_washing_machine + \
            hh_has_stove + hh_has_oven + hh_has_microwave + \
            food_as_part_of_wage_last_month + \
            dwelling_as_part_of_wage_last_month + \
            employers_transport_to_go_to_work + \
            goods_or_benefits_as_wage_last_month + \
            received_payments_from_interest_or_dividends_last_12_months + \
            stable_job + joined_any_trade_union + \
            satisfied_w_social_security_benefits + \
            had_other_job_before_current_one + \
            secondary_jobs_last_week + \
            willing_to_change_job + \
            compatible_working_schedule_and_family_responsibilities + \
            unpaid_work_last_week_raise_grow_animals + \
            unpaid_work_last_week_own_housework + \
            unpaid_work_last_week_childcare + \
            unpaid_work_last_week_training_courses + age + \
            passed_year_in_highest_degree + highest_education_title + \
            highest_degree + number_of_rooms + \
            number_of_hh_members + months_working_at_current_employment + \
            usual_weekly_hours_worked + months_worked_in_last_12_months', 
            #main_occupation_last_week_in_education + \
            #main_occupation_last_week_searching_for_work + \
            data=data_dict[year], return_type='dataframe')
        
            mod = qr(y, X)
            
            dif = int(year) - int(years[0])
            for i in range(0, dif+1): 
                aux_list = []
                for j, quantile in enumerate(quantiles):
                    aux_list.append(mod.predict(dict_params[str(int(year) - i)][j]))
               
                dict_predictions[year].append(aux_list) 
    
                # Calculate r-squareds and pseudo r-squareds for predictions 
                # with coefficient estimates from previous year
                for j, quantile in enumerate(quantiles):
                    # pseudo r-squared (copied from statsmodel 
                    # QuantRegResults.prsquared)
                    q = quantile
                    endog = data_dict[year]['ln_yearly_total_income_pc']
                    e = data_dict[year]['ln_yearly_total_income_pc'] - dict_predictions[year][i][j]
                    e = np.where(e < 0, (1 - q) * e, q * e)
                    e = np.abs(e)
                    ered = endog - stats.scoreatpercentile(endog, q * 100)
                    ered = np.where(ered < 0, (1 - q) * ered, q * ered)
                    ered = np.abs(ered)
                    pseudo_r_squareds.loc[(year, origin_coef[i]), 
                        columns[j]] = 1 - np.sum(e)/np.sum(ered)
    
                    # normal r-squared
                    res = np.sum(np.square(data_dict[year][
                        'ln_yearly_total_income_pc'] - dict_predictions[year][i][j]))
                    tot = np.sum(np.square(data_dict[year][
                        'ln_yearly_total_income_pc'] - data_dict[year][
                        'ln_yearly_total_income_pc'].mean()))
                    normal_r_squareds.loc[(year, origin_coef[i]), 
                        columns[j]] = 1 - res/tot
                    
                    # Calculate the mean squared error
                    mean_squared_errors.loc[(year, origin_coef[i]), 
                        columns[j]] = np.sum(np.square(
                        data_dict[year]['ln_yearly_total_income_pc'] - 
                        dict_predictions[year][i][j]))/len(data_dict[year])
                            
                    # Calculate the mean absolute error 
                    mean_abs_errors.loc[(year, origin_coef[i]), 
                        columns[j]] = np.sum(np.abs(
                        data_dict[year]['ln_yearly_total_income_pc'] - 
                        dict_predictions[year][i][j]))/len(data_dict[year])

        r_squareds = pd.concat({'pseudo_r_squareds': pseudo_r_squareds, 
                            'normal_r_squareds': normal_r_squareds}, axis=1)
            
    
        return dict_predictions, mean_squared_errors, r_squareds, mean_abs_errors#, dict_history_params


def ols_reg(years, data_dict, test_set, dict_params):
    """ Regress data from every survey year using statsmodels'
    OLS regression.
    Save coefficient estimates and then predict values of the 
    dependent variable every year with all its lagged coefficient 
    estimates and calculate r-squareds to test prediction power 
    of estimates over time. This means that the y's of every year 
    will have n_year - base_year predictions. Return a series with 
    the r-squareds of all predictions, a series with mean squared errors 
    of all predictions, a dictionary with all predictions and another 
    dictionary with the regression summaries for each year.
    
    Args:
        years: list with the survey years as strings
        data_dict: dictionary with data separated by survey year.
                The order of the variables is irrelevant
        test_set: if 'yes' then regress only on test data from random
            forest, if no, on training data
    
    """
    arrays = [['2015', '2016', '2016', '2017', '2017', '2017', 
               '2018', '2018', '2018', '2018', '2019', '2019',
               '2019', '2019', '2019'], 
          ['current', 'current', '-1', 'current', '-1', '-2', 
           'current', '-1', '-2', '-3', 'current', '-1', '-2',
           '-3', '-4']]
    multi_index = pd.MultiIndex.from_arrays(arrays, 
                        names=('year', 'origin_coefficients'))
    r_squareds = pd.Series(index=multi_index)
    mean_squared_errors = pd.Series(index=multi_index)
    mean_abs_errors = pd.Series(index=multi_index)
    
    dict_predictions = {year:[] for year in years}
    
    if test_set=='no':
        dict_params = {}
        for year in years: 
            y, X = dmatrices('ln_yearly_total_income_pc ~ male + \
            department_Atlántico + department_Bogotá + department_Bolívar + \
            department_Boyacá + department_Caldas + department_Caquetá + \
            department_Cauca + department_Cesar + department_Chocó + \
            department_Cundinamarca + department_Córdoba + department_Huila + \
            department_Guajira + department_Magdalena + department_Meta + \
            department_Nariño + department_Norte_de_Santander + \
            department_Quindio + department_Risaralda + department_Santander + \
            department_Sucre + department_Tolima + department_Valle_del_Cauca + \
            main_occupation_last_week_in_education + \
            main_occupation_last_week_searching_for_work + \
            main_occupation_last_week_working + \
            disease_expenses_coverage_hijos_o_familiares + \
            disease_expenses_coverage_seguro + \
            disease_expenses_coverage_subsidized + disease_expenses_coverage_eps + \
            disease_expenses_coverage_beneficiary + \
            disease_expenses_coverage_not_considered_yet + \
            disease_expenses_coverage_no_resources + \
            disease_expenses_coverage_borrowing_money + \
            overcrowding + health_insurance_type_special + \
            health_insurance_type_subsidized + \
            unpaid_work_last_week_elderly_or_handicap_care + \
            relationship_status_married + \
            relationship_status_not_married_and_less_than_two_year_relationship + \
            relationship_status_not_married_and_more_than_two_year_relationship + \
            relationship_status_single + relationship_status_widowed + \
            legal_waste_disposal + hh_has_vacuum_or_polisher + \
            hh_has_air_conditioning + hh_has_cable_tv + \
            hh_has_bicycle + hh_has_motorcycle + \
            hh_has_car + hh_has_holiday_home + \
            hh_has_internet + hh_has_washing_machine + \
            hh_has_stove + hh_has_oven + hh_has_microwave + \
            food_as_part_of_wage_last_month + \
            dwelling_as_part_of_wage_last_month + \
            employers_transport_to_go_to_work + \
            goods_or_benefits_as_wage_last_month + \
            received_payments_from_interest_or_dividends_last_12_months + \
            stable_job + joined_any_trade_union + \
            satisfied_w_social_security_benefits + \
            had_other_job_before_current_one + \
            secondary_jobs_last_week + \
            willing_to_change_job + \
            compatible_working_schedule_and_family_responsibilities + \
            unpaid_work_last_week_raise_grow_animals + \
            unpaid_work_last_week_own_housework + \
            unpaid_work_last_week_childcare + \
            unpaid_work_last_week_training_courses + age + \
            passed_year_in_highest_degree + highest_education_title + \
            highest_degree + number_of_rooms + \
            number_of_hh_members + months_working_at_current_employment + \
            usual_weekly_hours_worked + months_worked_in_last_12_months', 
            data=data_dict[year], return_type='dataframe')
                
            mod = sm.OLS(y, X)
            result = mod.fit() 
            dict_params[year] = result.params
            
        return dict_params
                
    else:
        for year in years:
            y, X = dmatrices('ln_yearly_total_income_pc ~ male + \
            department_Atlántico + department_Bogotá + department_Bolívar + \
            department_Boyacá + department_Caldas + department_Caquetá + \
            department_Cauca + department_Cesar + department_Chocó + \
            department_Cundinamarca + department_Córdoba + department_Huila + \
            department_Guajira + department_Magdalena + department_Meta + \
            department_Nariño + department_Norte_de_Santander + \
            department_Quindio + department_Risaralda + department_Santander + \
            department_Sucre + department_Tolima + department_Valle_del_Cauca + \
            main_occupation_last_week_in_education + \
            main_occupation_last_week_searching_for_work + \
            main_occupation_last_week_working + \
            disease_expenses_coverage_hijos_o_familiares + \
            disease_expenses_coverage_seguro + \
            disease_expenses_coverage_subsidized + disease_expenses_coverage_eps + \
            disease_expenses_coverage_beneficiary + \
            disease_expenses_coverage_not_considered_yet + \
            disease_expenses_coverage_no_resources + \
            disease_expenses_coverage_borrowing_money + \
            overcrowding + health_insurance_type_special + \
            health_insurance_type_subsidized + \
            unpaid_work_last_week_elderly_or_handicap_care + \
            relationship_status_married + \
            relationship_status_not_married_and_less_than_two_year_relationship + \
            relationship_status_not_married_and_more_than_two_year_relationship + \
            relationship_status_single + relationship_status_widowed + \
            legal_waste_disposal + hh_has_vacuum_or_polisher + \
            hh_has_air_conditioning + hh_has_cable_tv + \
            hh_has_bicycle + hh_has_motorcycle + \
            hh_has_car + hh_has_holiday_home + \
            hh_has_internet + hh_has_washing_machine + \
            hh_has_stove + hh_has_oven + hh_has_microwave + \
            food_as_part_of_wage_last_month + \
            dwelling_as_part_of_wage_last_month + \
            employers_transport_to_go_to_work + \
            goods_or_benefits_as_wage_last_month + \
            received_payments_from_interest_or_dividends_last_12_months + \
            stable_job + joined_any_trade_union + \
            satisfied_w_social_security_benefits + \
            had_other_job_before_current_one + \
            secondary_jobs_last_week + \
            willing_to_change_job + \
            compatible_working_schedule_and_family_responsibilities + \
            unpaid_work_last_week_raise_grow_animals + \
            unpaid_work_last_week_own_housework + \
            unpaid_work_last_week_childcare + \
            unpaid_work_last_week_training_courses + age + \
            passed_year_in_highest_degree + highest_education_title + \
            highest_degree + number_of_rooms + \
            number_of_hh_members + months_working_at_current_employment + \
            usual_weekly_hours_worked + months_worked_in_last_12_months', 
            data=data_dict[year], return_type='dataframe')
            
            mod = sm.OLS(y, X)
            
            dif = int(year) - int(years[0])
            for i in range(0, dif+1): 
                dict_predictions[year].append(mod.predict(
                    dict_params[str(int(year) - i)]))
                
                # Calculate r-squareds for predictions with 
                # coefficient estimates from previous year
                res = np.sum(np.square(data_dict[year][
                    'ln_yearly_total_income_pc'] - dict_predictions[year][i]))
                tot = np.sum(np.square(data_dict[year][
                    'ln_yearly_total_income_pc'] - data_dict[year][
                    'ln_yearly_total_income_pc'].mean()))
                r_squareds[year][i] = 1 - res/tot
                
                # Calculate the mean squared error
                mean_squared_errors[year][i] = np.sum(np.square(
                    data_dict[year]['ln_yearly_total_income_pc'] - 
                    dict_predictions[year][i]))/len(data_dict[year])
            
                # Calculate the mean absolute error
                mean_abs_errors[year][i] = np.sum(np.abs(
                    data_dict[year]['ln_yearly_total_income_pc'] - 
                    dict_predictions[year][i]))/len(data_dict[year])
                     
        return dict_predictions, mean_squared_errors, r_squareds, mean_abs_errors