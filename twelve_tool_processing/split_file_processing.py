#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.preprocessing import scale
pd.set_option('display.width',500)

# 将异常值置换为空
def replace_outlier_nan(df, outlier_dic):
    columns = df.columns
    outlier_col = list(outlier_dic.keys())
    dic = {}
    count = 0
    for i in columns: #对df的列名做编号
        dic[i] = count
        count += 1
    # 将df转为np，进行修改
    array = np.array(df)
    n_rows = array.shape[0]
    for i in outlier_col:
        j = dic[i] # j表示在异常值列的第几列
        for out_v in outlier_dic[i]: #遍历某列异常值集合
            for k in range(n_rows):  #遍历数组行
                if np.abs(array[k][j]-out_v) < 0.00001:
                    array[k][j] = np.NaN
    df = pd.DataFrame(array, columns=columns)
    return df



# 填补缺失值
def fill_NaN(df,threld_category):
    for i in df.columns:
        s = df[i]
        if len(pd.value_counts(s)) < threld_category:
            imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        else:
            imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(df)
    df1 = imp.transform(df)
    df2=pd.DataFrame(df1,columns=df.columns)
    return df2


# 将异常值0替换为缺失值
def deal_zero(df,median_thd):
    lower_bool = df.apply(lambda x: x.min() == 0 and x.median() > median_thd)
    df.loc[:, df.columns[lower_bool]] = df.loc[:, df.columns[lower_bool]].replace(0,np.nan)
    upper_bool = df.apply(lambda x: x.max() == 0 and x.median() < -median_thd)
    df.loc[:, df.columns[upper_bool]] = df.loc[:, df.columns[upper_bool]].replace(0,np.nan)
    return df

# 剔除众数比例大于阈值的属性
def drop_frequent(df,frequent_thd):
    frequent_column=[]
    for i in df.columns:
        s = df[i]
        if float((pd.value_counts(s)).max()) / float(len(s)) > frequent_thd:
            frequent_column.append(i)
    df=df.drop(frequent_column,axis=1)
    return df

# 相同量级的相邻列作差
def dealta_same_metric(df):
    augmented_df = df
    for i in range(df.shape[1] - 1):
        first_col = df.iloc[:, i]
        last_col = df.iloc[:, i + 1]
        if  last_col.mean() / first_col.mean() > 10 or last_col.mean() / first_col.mean() < 0.1:
            continue
        else:
            augmented_df = pd.concat([augmented_df, pd.Series(last_col - first_col, name=last_col.name + '-' + first_col.name)],axis=1)
    return augmented_df.drop(df,axis=1)

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
            minus = k[12:]
            if year == '2016':
                k = str(train_data_date[i][j-1])
            if minute == '60':
                minute = '00'
                hour_new = str(int(hour) + 1)
                if len(hour_new) == 1:
                    hour_new = '0' + hour_new
                k = year + month + day + hour_new + minute + minus
            date_data.append(k)
    date_data = np.array(date_data).reshape(train_data_date.shape[0], train_data_date.shape[1])

    date_data = pd.DataFrame(date_data, columns=df_date.columns)
    near_date_data = deal_time_metric( date_data).drop(date_columns,axis=1)

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
    all_delta_time = pd.DataFrame(delta_time, columns=['time_delta_0'])
    new_time=pd.concat([near_date_data,all_delta_time],axis=1)
    return new_time

# 特征选择:pearsonr系数
def select_feature_pearsonr(train_data,train_data_y):
    corr_column = []
    for i in train_data.columns:
        corr = pearsonr(train_data[i], train_data_y)
        r = abs(corr[0])
        p = corr[1]
        if p < 0.05 and r > 0.202:
            corr_column.append(i)
    return corr_column

def select_categorical(data, category_n):
    discrete_columns = []
    for i in data.columns:
        if len(pd.value_counts(data[i].dropna())) <= category_n:
            discrete_columns.append(i)
    return discrete_columns


def train_data_discrete_to_one_hot(df, tool_id_list):
    # 对数字类别做编码
    for i in tool_id_list:
        df = pd.concat([df, pd.get_dummies(df[i], prefix=i)], axis=1)
        df.drop(i, axis=1, inplace=True)
    return df

if __name__=="__main__":
    # train_data_y=pd.read_csv('C:/Users/user/Desktop/train_data_y.csv',header=None)

    # train_data_a = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/train.xlsx')
    # train_data_y = train_data_a.iloc[:, -1]
    # print train_data_y

    TOOL1_data = pd.read_csv('C:/Users/user/Desktop/tool_testA_data/TOOL1.csv').drop('TOOL',axis=1)
    date_columns=['210X24', '210X204', '210X205', '210X213', '210X215']

    # 挑出时间特征
    new_date_columns=deal_time(TOOL1_data,date_columns)
    new_date_columns.to_csv('C:/Users/user/Desktop/file_testA/tool_1.csv', index=False)

    # 训练集预处理
    # train_data = TOOL1_data.drop(date_columns,axis=1)
    # train_data = deal_zero(train_data, median_thd=0.2)
    # train_data = train_data.dropna(axis=1, how='any', thresh=TOOL1_data.shape[0] * 0.6)
    # train_data = fill_NaN(train_data, threld_category=6)
    #
    #
    # train_data=drop_frequent(train_data,frequent_thd=0.95)
    # train_data = train_data.T.drop_duplicates().T
    #
    # # 挑出相同量级相邻特征之差
    # neighbor_feature = dealta_same_metric(train_data)
    # train_data=pd.concat([train_data,new_date_columns,neighbor_feature],axis=1)
    #
    # corr_column=select_feature_pearsonr(train_data,train_data_y)
    # train_data=train_data[corr_column]
    # print train_data.shape
    #
    # discrete_columns = select_categorical(train_data, category_n=5)
    # train_data_discrete = train_data[discrete_columns]
    #
    # train_data_discrete_dummies = train_data_discrete_to_one_hot(train_data_discrete, discrete_columns)
    # train_data_sequent = train_data.drop(discrete_columns, axis=1)
    #
    # train_data_new=pd.concat([train_data_sequent,train_data_discrete_dummies],axis=1)
    # print train_data_new.shape
    # train_data_new.to_csv('C:/Users/user/Desktop/file_train/tool_1.csv',index=False)

