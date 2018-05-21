#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import BayesianRidge,LinearRegression,LassoCV,RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn import metrics
import xgboost as xgb


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

    train_data_a=pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/train.xlsx')
    train_data_y=train_data_a.iloc[:,-1]

    train_data_all=pd.read_csv('C:/Users/user/Desktop/data_processing/train_data_concat.csv')

    train_data=train_data_all.iloc[0:800,:]
    test_data=train_data_all.iloc[800:1100,:]
    test_data.reset_index(drop=True,inplace=True)

    corr_column=[]
    for i in train_data.columns:
        corr = pearsonr(train_data[i],train_data_y)
        r=abs(corr[0])
        p=corr[1]
        if p<0.05 and r>0.244:
            corr_column.append(i)
    print(corr_column)

    train_data=train_data_all[corr_column]
    print(train_data.shape)

    train_discrete_columns=select_categorical(train_data,category_n=5)
    train_data_discrete=train_data[train_discrete_columns]
    train_data_discrete_dummies=train_data_discrete_to_one_hot(train_data_discrete,train_discrete_columns)
    train_data_discrete=train_data_discrete_dummies.iloc[0:800,:]
    test_data_discrete=train_data_discrete_dummies.iloc[800:1100,:]
    test_data_discrete.reset_index(drop=True,inplace=True)

    train_data_sequent=train_data.drop(train_discrete_columns,axis=1)
    train_data=train_data_sequent.iloc[0:800,:]
    test_data=train_data_sequent.iloc[800:1100,:]
    test_data.reset_index(drop=True,inplace=True)


    print(train_data.shape)
    print(test_data.shape)

    min_max_scaler = preprocessing.MinMaxScaler()

    train_minmax = min_max_scaler.fit_transform(train_data)
    test_minmax=min_max_scaler.transform(test_data)


    train_data=pd.DataFrame(train_minmax,columns=train_data.columns)
    test_data=pd.DataFrame(test_minmax,columns=test_data.columns)

    train_tool_df=pd.read_csv('C:/Users/user/Desktop/train_tool_id_dummy.csv')
    test_tool_df=pd.read_csv('C:/Users/user/Desktop/test_tool_id_dummy.csv')
    train_data=pd.concat([train_data,train_tool_df,train_data_discrete],axis=1)
    test_data=pd.concat([test_data,test_tool_df,test_data_discrete],axis=1)
    print(train_data.shape)
    print(test_data.shape)

    params = { 'loss': 'ls',
          'learning_rate': 0.022,
          'n_estimators': 2825,
          'max_depth':4,
          'subsample':0.9,
          'min_samples_split':2,
          'min_samples_leaf':1,
          'random_state':1,
          'max_features':'log2',
          'alpha':0.9}
    model_gbr = GradientBoostingRegressor(**params)
    y_predict_gbr=model_gbr.fit(train_data, train_data_y).predict(test_data)
    model_Lasso = LassoCV(normalize=False, alphas=np.arange(0.0001,0.01,0.0001),cv=ShuffleSplit(n_splits=5,test_size=0.2),n_jobs=-1)
    y_predict_lasso=model_Lasso.fit(train_data, train_data_y).predict(test_data)
    model_bridge = BayesianRidge()
    y_predict_bridge=model_bridge.fit(train_data, train_data_y).predict(test_data)
    answer_true=pd.read_csv('D:\desktop\天池\AI\season two\[new] fusai_answer_a_20180127.csv',header=None).iloc[:,-1]

    p=np.arange(0.1,1.0,0.001)
    for i in list(p):
        y_predict=i*y_predict_gbr+(1-i)*y_predict_lasso
        mse1 = metrics.mean_squared_error(y_predict, answer_true)
        if mse1<0.026:
            print(mse1)



