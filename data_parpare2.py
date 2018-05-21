#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np
pd.set_option('display.width',500)

# 训练集做四步预处理   测试集做一步预处理
def data_processing(train_data,test_data,thresh_NA_rating):
    ID_list = ['ID','TOOL', 'Tool', 'TOOL_ID', 'Tool (#1)', 'TOOL (#1)', 'TOOL (#2)', 'Tool (#2)', 'Tool (#3)',
               'Tool (#4)', 'OPERATION_ID', 'Tool (#5)', 'TOOL (#3)']
    train_data.drop(ID_list, axis=1, inplace=True)
    test_data.drop(ID_list, axis=1, inplace=True)
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
    test_data.drop(date_list, axis=1, inplace=True)

    train_data=train_data.dropna(axis=1, how='any', thresh=train_data.shape[0] * thresh_NA_rating)
    test_data = test_data.dropna(axis=1, how='any', thresh=test_data.shape[0] * thresh_NA_rating)

    train_data=train_data.T.drop_duplicates().T

    std_series_train = train_data.std()
    for i in std_series_train.index:
        if std_series_train[i] < 0.00001:
            train_data.drop(i, axis=1, inplace=True)

    test_columns = set(test_data.columns)
    train_columns = set(train_data.columns)
    b = list(test_columns & train_columns)  # 求两者的交集，转化为列表

    train_data = train_data[b]
    test_data = test_data[b]
    train_data.to_csv('C:/Users/user/Desktop/data2_processing/train_data.csv', index=False)
    test_data.to_csv('C:/Users/user/Desktop/data2_processing/test_data.csv', index=False)
    print train_data.shape
    print test_data.shape


if __name__=="__main__":
    train_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/train.xlsx')
    testA_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/testA.xlsx')
    data_processing(train_data,testA_data,thresh_NA_rating=0.6)





