#!/usr/bin/python
#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
pd.set_option('display.width',500)
train_data=pd.read_csv('C:/Users/user/Desktop/testB_tool_data/Tool (#5)11.csv')
# for i in range(train_data['210X24'].shape[0]):
#     if train_data['210X24'][i]==2.01666166617e+13:
#        train_data['210X24'][i]=np.NaN
train_data_date1=train_data[[ '520X148', '520X152', '520X171', '520X173', '520X248', '520X250', '520X346', '520X348', '520X354', '520X356']]
train_data_date= train_data_date1.fillna(method='ffill')
train_data_date = train_data_date.fillna(method='bfill')
# print train_data_date
train_data_date = np.array(train_data_date.astype('int64'))
date_data=[]
for i in range(train_data_date.shape[0]):
    for j in range(train_data_date.shape[1]):
        k= str(train_data_date[i][j])
        year=k[0:4]
        month=k[4:6]
        day=k[6:8]
        hour=k[8:10]
        minute=k[10:12]
        minus=k[12:14]
        if year=='2016':
            k=np.NaN
        if minute=='60':
            minute='00'
            hour_new=str(int(hour)+1)
            if len(hour_new)==1:
                hour_new='0'+hour_new
            k=year+month+day+hour_new+minute+minus
        k=year+month+day+hour+minute+minus
        date_data.append(k)
date_data=np.array(date_data).reshape(train_data_date.shape[0],train_data_date.shape[1])

date_data=pd.DataFrame(date_data,columns=train_data_date1.columns)
TOOL1_date=date_data.astype('float64')

TOOL1_date['max_date']=TOOL1_date.T.max()
TOOL1_date['min_date']=TOOL1_date.T.min()

min_value=TOOL1_date['min_date'].astype('int64')
max_value=TOOL1_date['max_date'].astype('int64')
max_date=[]
min_date=[]
delta_time=[]
count=0
for i in range(len(max_value)):
    min_str = str(min_value[i])
    max_str=  str(max_value[i])
    min_str = datetime.strptime(min_str, '%Y%m%d%H%M%S')
    max_str = datetime.strptime(max_str, '%Y%m%d%H%M%S')
    time_delta = max_str-min_str
    count+=1
    # print min_str,count
    min_date.append(min_str)
    max_date.append(max_str)
    delta_time.append(time_delta.seconds)
max_date=pd.DataFrame(max_date,columns=['max_date_TOOL1'])
min_date=pd.DataFrame(min_date,columns=['min_date_TOOL1'])
delta_time=pd.DataFrame(delta_time,columns=['time_delta_5'])
df=pd.concat([max_date,min_date,delta_time],axis=1)
df.to_csv('C:/Users/user/Desktop/testB_deal_date/tool_time_delta_5.csv',index=False)

