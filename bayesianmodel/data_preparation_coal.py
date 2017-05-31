# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:59:49 2017

@author: Administrator
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:/Users/Administrator/Documents/bayesianmodel/data')

date_parse = lambda x: pd.datetime.strptime(x.replace('/','-'), '%Y-%m-%d')

df_day_coal = pd.read_csv('data_day_frequency_coal.csv', index_col = 'date', date_parser = date_parse)
df_day_target = pd.read_csv('bltyy_price.csv', index_col = 'date', date_parser = date_parse)
df_week_iron = pd.read_csv('lwgkc.csv', index_col = 'date', date_parser = date_parse)
df_week_rawiron = pd.read_csv('gtqycy_kc.csv', index_col = 'date', date_parser = date_parse)

statis_null = df_day_coal.isnull().sum()
statis_null = pd.DataFrame(statis_null)
index_drop = statis_null[statis_null[0]>=500].index.tolist()

for col in index_drop:
    df_day_coal.drop(col, axis = 1, inplace = True)

statis_null = df_day_coal.isnull().sum()

for col in df_day_coal.columns:
    temp = df_day_coal[col].value_counts().sort_values(ascending = False)
    if temp[temp.index[0]] >= 500:
         df_day_coal.drop(col, axis = 1, inplace = True)
#df_day_coal.dropna(inplace = True)

for col in df_day_coal.columns:
    df_day_coal[col].fillna(method = 'bfill', inplace = True)

df_day_coal.dropna(inplace = True)

df_day_iron = df_week_iron.resample('D',fill_method = 'ffill').sort_index(ascending = False)
df_day_rawiron = df_week_rawiron.resample('D',fill_method = 'ffill').sort_index(ascending = False)

df_day_full = df_day_coal.join([df_day_iron,df_day_rawiron,df_day_target])

df_day_full.fillna(method = 'bfill', inplace = True)

#df_day_full.to_csv('data_day_fullyprepared.csv')