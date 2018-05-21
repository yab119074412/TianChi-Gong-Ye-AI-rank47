#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import pearsonr
from datetime import datetime
pd.set_option('display.width',500)


# 相邻时间列作差
def deal_time_metric(df):
    augmented_df = df
    for i in xrange(df.shape[1] - 1):
        first_col = df.iloc[:, i]
        last_col = df.iloc[:, i + 1]
        if len(last_col[0]) / len(first_col[0]) > 10 or len(last_col[0]) / len(first_col[0]) < 0.1:
            continue
        else:
            time_delta=[]
            for j in range(len(first_col)):
                one_col = datetime.strptime(first_col[j], '%Y%m%d%H%M%S')
                two_col = datetime.strptime(last_col[j], '%Y%m%d%H%M%S')
                if one_col>two_col:
                    c=one_col
                    one_col=two_col
                    two_col=c
                delta_time= (two_col - one_col).seconds
                time_delta.append(delta_time)
                time1_delta=pd.Series(time_delta, name=last_col.name + '-' + first_col.name)
            augmented_df = pd.concat([augmented_df, time1_delta], axis=1)
    return augmented_df


# 机台制成零件所需时间
def deal_time(df,date_columns):
    df_date=df[date_columns]
    train_data_date = df_date.fillna(method='ffill')
    train_data_date = train_data_date.fillna(method='bfill')

    train_data_date = np.array(train_data_date.astype('int64'))
    date_data = []
    for i in range(train_data_date.shape[0]):
        for j in range(train_data_date.shape[1]):
            k = str(train_data_date[i][j])
            year = k[0:4]
            month = k[4:6]
            day = k[6:8]
            hour = k[8:10]
            minute = k[10:12]
            minus = k[12:14]
            if year == '2016':
                k = str(train_data_date[i][j-1])
            if minute == '60':
                minute = '00'
                hour_new = str(int(hour) + 1)
                if len(hour_new) == 1:
                    hour_new = '0' + hour_new
                k = year + month + day + hour_new + minute + minus
            k = year + month + day + hour + minute + minus
            date_data.append(k)
    date_data = np.array(date_data).reshape(train_data_date.shape[0], train_data_date.shape[1])

    date_data = pd.DataFrame(date_data, columns=df_date.columns)
    near_date_data = deal_time_metric(date_data).drop(date_columns, axis=1)

    TOOL1_date = date_data.astype('float64')

    TOOL1_date['max_date'] = TOOL1_date.T.max()
    TOOL1_date['min_date'] = TOOL1_date.T.min()

    min_value = TOOL1_date['min_date'].astype('int64')
    max_value = TOOL1_date['max_date'].astype('int64')
    delta_time = []
    for i in range(len(max_value)):
        min_str = str(min_value[i])
        max_str = str(max_value[i])
        min_str = datetime.strptime(min_str, '%Y%m%d%H%M%S')
        max_str = datetime.strptime(max_str, '%Y%m%d%H%M%S')
        time_delta = max_str - min_str
        delta_time.append(time_delta.seconds)
    all_delta_time = pd.DataFrame(delta_time, columns=['time_delta_3'])
    new_time=pd.concat([near_date_data,all_delta_time],axis=1)
    return new_time

if __name__=="__main__":

    TOOL1_data = pd.read_csv('C:/Users/user/Desktop/tool_testA_data/Tool (#1)4.csv').drop('Tool (#1)',axis=1)
    date_columns=[ '310X56', '310X60', '310X64', '310X68', '310X72', '310X76', '310X80', '310X84','311X6', '311X7', '311X20', '311X22',
                              '311X55', '311X56','311X59', '311X60', '311X78', '311X79', '311X163', '311X164', '311X170', '311X171']

    # 挑出时间特征
    new_date_columns=deal_time(TOOL1_data,date_columns)
    # print new_date_columns
    new_date_columns.to_csv('C:/Users/user/Desktop/file_testA/tool_4.csv', index=False)


