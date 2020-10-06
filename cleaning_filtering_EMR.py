#!/usr/bin/env python
# coding: utf-8

# import libraries 
import csv
import sys
import numpy as np
import pandas as pd 
import os
from datetime import datetime
from statistics import mean, median, mode, stdev
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import nltk
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import chi2_contingency
from statsmodels.formula.api import ols
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import Lasso 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import r2_score

filepath1 = "~/Desktop/REDCap.csv"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def open_file_create_df(filepath):
    # opening file and creating file handler 
    file_handler = open(os.path.expanduser(filepath))
    # creating pandas dataframe 
    df = pd.read_csv(file_handler, sep = ",") 
    return df

dataframe = open_file_create_df(filepath1)

def clean_and_filter_phq9(dataframe):
    # create copy of dataframe passed in as a parameter 
    df_phq = dataframe.copy()
    # extract only the record id, result date, phq score, and phq type
    df_phq = df_phq[["record_id","mr_v1_phq_date", "mr_v1_phq_score", "mr_v1_phq"]]
    # filter to retrieve the total phq9 score 
    filterForPHQ9 = df_phq['mr_v1_phq'].isin(['phq9_total'])
    # create a dataframe for the phq9 score, result date, and record id
    df_phq9 = df_phq.copy()
    df_phq9 = df_phq9[filterForPHQ9]
    df_phq9.reset_index(drop=True, inplace=True)
    df_phq9 = df_phq9.drop(["mr_v1_phq"], axis=1)
    # sort the dataframe from earliest datatime 
    df_phq9 = filter_earliest_datetime(df_phq9, "mr_v1_phq_date", "record_id")
    df_phq9 = df_phq9.sort_values(by ="record_id" )
    nan_value = float("NaN")
    df_phq9.replace("", nan_value, inplace=True)
    # handling missing data by dropping patients that don't have a complete phq9
    df_phq9.dropna(subset = ["mr_v1_phq_score"], inplace=True)
    df_phq9.reset_index(drop=True, inplace=True)
    df_phq9 = df_phq9.drop(["mr_v1_phq_date"], axis=1)    
    return df_phq9
    
date_var = 'mr_v1_phq_date'

def filter_earliest_datetime(dataframe, date_var, id_var):
    # sort based on datetime
    dataframe[date_var] = dataframe[date_var].astype('datetime64[ns]')
    dataframe = dataframe.sort_values(by = date_var)
    # delete the duplicates and keep first instance
    dataframe.drop_duplicates(subset = id_var, 
                     keep = 'first', inplace = True)
    return dataframe 


def get_demo(dataframe, demo):
    df_demo = dataframe.copy()
    # create dataframe with just the specified demo 
    df_demo = df_demo[["record_id", demo]]
    # handle duplicate values and keep the first instance 
    df_demo.drop_duplicates(subset ="record_id", 
                         keep = "first", inplace = True)
    df_demo.reset_index(drop=True, inplace=True)
    df_demo[demo] = df_demo[demo].astype(float)
    return df_demo


def get_labs(dataframe):
    df_labs = dataframe.copy()
    # create a dataframe for the record id, lab name, result date, and result value
    df_labs = df_labs[["record_id","mr_v1_lab_name","mr_v1_result_date", "mr_v1_result_value"]]
    nan_value = float("NaN")
    df_labs.replace("", nan_value, inplace=True)
    # delete NaN lab name entries (handles spaces in data)
    df_labs.dropna(subset = ["mr_v1_lab_name"], inplace=True)
    df_labs.reset_index(drop=True, inplace=True)
    return df_labs

def get_specific_lab(dataframe, lab_name):
    # filter for the specified lab
    filtered_lab = dataframe['mr_v1_lab_name'].isin([lab_name])
    df_lab = dataframe.copy()
    df_lab = df_lab[filtered_lab]
    # sort the dataframe from earliest datatime 
    df_lab = filter_earliest_datetime(df_lab, 'mr_v1_result_date', "record_id")
    df_lab = df_lab.sort_values(by ="record_id" )
    df_lab.reset_index(drop=True, inplace=True)
    # clean the dataframe - rename and drop unnecessary columns 
    df_lab.rename(columns={"mr_v1_result_value":lab_name}, inplace = True)
    df_lab = df_lab.drop(["mr_v1_lab_name", "mr_v1_result_date"], axis=1)
    df_lab[lab_name] = df_lab[lab_name].astype(float)
    return df_lab


# get the phq9 dataframe  
df_phq9 = clean_and_filter_phq9(dataframe)
# get the labs dataframe 
labs = get_labs(dataframe)
# get the microbiome redcap data 
filepath2 = "~/Desktop/microbiome.csv"
dataframe_mb = open_file_create_df(filepath2)
# missing data handling 
dataframe_mb.dropna(subset = ["visit_visittype"], inplace=True)
dataframe_mb = dataframe_mb[dataframe_mb.record_id.apply(lambda x: x.isnumeric())]
dataframe_mb['record_id'] = dataframe_mb['record_id'].astype(int)

# the variables of interest from the ccsts and microbiome project 
categorical_ccts = ['mr_v1_ethnicity','mr_v1_race', 'mr_v1_insurance', 'pi_age']
continuous_ccts = ['wbc','rbc','1hr_gluc_challenge', 'plt', 'rdw', 'mpv',
              'hct', 'hgb', 'mchc', 'mch', 'mcv', '%lymph', '%neut', '%monos', '%baso', '%eos']

categorical_microbiome = ['demo_income','demo_work', 'demo_education', 'demo_totalchildren', 'mr_v1_bmi_initialob']
continuous_microbiome = []

def create_features_dataframe(dataframe, categorical, continuous):
    # since the merge function needs to have only two dataframes 
    features = get_demo(dataframe, categorical[0])
    categorical.remove(categorical[0])
    # merging demographics and labs into one dataframe 
    for i in categorical:
        i = get_demo(dataframe, i)
        features = pd.merge(features, i, on="record_id", how='left')
    for i in continuous:
        i = get_specific_lab(labs, i)
        features = pd.merge(features, i, on="record_id", how='left')
    return features

features = create_features_dataframe(dataframe, categorical_ccts, continuous_ccts)
# merging the dataframes from both projects 
features = pd.merge(features, create_features_dataframe(dataframe_mb, categorical_microbiome, continuous_microbiome), on="record_id", how='left')


