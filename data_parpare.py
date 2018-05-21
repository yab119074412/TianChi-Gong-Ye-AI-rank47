#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

pd.set_option('display.width',500)

# 将异常值置换为空
def replace_outlier_nan(df, outlier_dic_train):
    columns = df.columns
    outlier_col = list(outlier_dic_train.keys())
    dic_train = {}
    count = 0
    for i in columns: #对df的列名做编号
        dic_train[i] = count
        count += 1
    # 将df转为np，进行修改
    array = np.array(df)
    n_rows = array.shape[0]
    for i in outlier_col:
        j = dic_train[i] # j表示在异常值列的第几列
        for out_v in outlier_dic_train[i]: #遍历某列异常值集合
            for k in range(n_rows):  #遍历数组行
                if np.abs(array[k][j]-out_v) < 0.00001:
                    array[k][j] = np.NaN
    df = pd.DataFrame(array, columns=columns)
    return df


# 处理0值异常，(借鉴别人的)
def deal_zero(df,median_thd):
    lower_bool = df.apply(lambda x: x.min() == 0 and x.median() > median_thd)
    df.loc[:, df.columns[lower_bool]] = df.loc[:, df.columns[lower_bool]].replace(0,np.nan)
    upper_bool = df.apply(lambda x: x.max() == 0 and x.median() < -median_thd)
    df.loc[:, df.columns[upper_bool]] = df.loc[:, df.columns[upper_bool]].replace(0,np.nan)
    return df

# 填补缺失值
def fill_NaN(df,threld_category):
    for i in df.columns:
        s = df[i]
        if len(pd.value_counts(s)) <= threld_category:
            imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
        else:
            imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    imp.fit(df)
    df1 = imp.transform(df)
    df2=pd.DataFrame(df1,columns=df.columns)
    return df2

# 相同量级的相邻列作差
def dealta_same_metric(df):
    augmented_df = df
    for i in range(df.shape[1] - 1):
        first_col = df.iloc[:, i]
        last_col = df.iloc[:, i + 1]
        if  (first_col.mean()!=0 or first_col.mean()!=0) and (last_col.mean() / first_col.mean() > 10 or last_col.mean() / first_col.mean() < 0.1):
            continue
        else:
            augmented_df = pd.concat([augmented_df, pd.Series(last_col - first_col, name=last_col.name + '-' + first_col.name)],axis=1)
    return augmented_df

def get_dic(df):
    dic_train = {}
    for i_col in df.columns:
        s = df[i_col].dropna()
        if len(pd.value_counts(s)) <= 10:
            median = s.median()
            if median == 0:
                median = s.mean()
            num_except_null = len(s)
            value_count = pd.value_counts(s)
            value_count = value_count / (num_except_null / len(value_count))
            for i_vc in value_count.index:
                if value_count[i_vc] < 0.03 and np.abs((i_vc - median) / median) > 0.9 and np.abs(
                                np.abs(i_vc) - np.abs(median)) > 0.2:
                    if i_col in dic_train:
                        dic_train[i_col].add(i_vc)
                    else:
                        dic_train[i_col] = set()
                        dic_train[i_col].add(i_vc)
    return dic_train


# 训练集、测试集同时做四步预处理
def data_processing(train_data,test_data,thresh_NA_rating):
    train_data = pd.concat([train_data, test_data], axis=0)
    train_data.reset_index(drop=True, inplace=True)
    ID_list = ['ID','TOOL', 'Tool', 'TOOL_ID', 'Tool (#1)', 'TOOL (#1)', 'TOOL (#2)', 'Tool (#2)', 'Tool (#3)',
               'Tool (#4)', 'OPERATION_ID', 'Tool (#5)', 'TOOL (#3)']
    train_data.drop(ID_list, axis=1, inplace=True)
    # test_data.drop(ID_list, axis=1, inplace=True)
    date_list = ['210X24', '210X204', '210X205', '210X213', '210X215', '220X67', '220X71', '220X75', '220X79', '220X83',
                 '220X87', '220X91', '220X95', '300X2', '300X3',
                 '300X4', '300X6', '300X7', '300X9', '300X10', '300X13', '300X14', '300X20', '310X56', '310X60',
                 '310X64', '310X68', '310X72', '310X76', '310X80', '310X84',
                 '311X6', '311X7', '311X20', '311X22', '311X55', '311X56', '311X59', '311X60', '311X78', '311X79',
                 '311X163', '311X164', '311X170', '311X171', '330X640',
                 '330X641', '330X1165', '330X1168', '330X1169', '360X710', '360X711', '360X1287', '360X1291',
                 '360X1292', '360X1293', '400X7', '400X9', '400X25', '400X27',
                 '400X60', '400X61', '400X64', '400X65', '400X83', '400X84', '400X168', '400X169', '400X219', '400X220',
                 '420X7', '420X9', '420X25', '420X27', '520X148',
                 '520X152', '520X171', '520X173', '520X248', '520X250', '520X346', '520X348', '520X354', '520X356',
                 '750X710', '750X711', '750X1287', '750X1291',
                 '750X1292', '750X1293']
    train_data.drop(date_list, axis=1, inplace=True)

    train_data = train_data.T.drop_duplicates().T

    train_data = train_data.dropna(axis=1, how='any', thresh=train_data.shape[0] * thresh_NA_rating)

    # train_data = train_data.T.drop_duplicates().T

    std_series_train = train_data.std()
    for i in std_series_train.index:
        if std_series_train[i] < 0.00001:
            train_data.drop(i, axis=1, inplace=True)

    train_data = deal_zero(train_data, median_thd=0.2)
    # 对离散数据做分析，
    dic_train = get_dic(train_data)
    train_data = replace_outlier_nan(train_data, dic_train)

    train_data = fill_NaN(train_data, 10)

    # train_data = drop_frequent(train_data, frequent_thd=0.95)
    # test_data = drop_frequent(test_data, frequent_thd=0.95)

    # neighbor_feature_train= dealta_same_metric(train_data)
    # neighbor_feature_test=dealta_same_metric(test_data)
    # train_data.to_csv('C:/Users/user/Desktop/data_processing/neighbor_feature_train.csv', index=False)
    # test_data.to_csv('C:/Users/user/Desktop/data_processing/test_data.csv', index=False)


    print train_data.shape
    train_data.to_csv('C:/Users/user/Desktop/data_processing/train_data_concat.csv', index=False)

if __name__=="__main__":
    train_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/train.xlsx').drop('Value',axis=1)
    testA_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/testA.xlsx')
    train_data_all=pd.concat([train_data,testA_data],axis=0)
    train_data_all.reset_index(drop=True,inplace=True)

    testB_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/testB.xlsx')
    data_processing(train_data,testB_data,thresh_NA_rating=0.6)



