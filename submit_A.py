#!/usr/bin/python
#-*- coding:gbk -*-
import pandas as pd
import numpy as np

if __name__=="__main__":

    testB_data = pd.read_excel('D:/jupyterNotebook/AI2.0/data_csv/testB.xlsx')

    ID_list=['TOOL','Tool','TOOL_ID','Tool (#1)','TOOL (#1)','TOOL (#2)','Tool (#2)','Tool (#3)','Tool (#4)','OPERATION_ID','Tool (#5)','TOOL (#3)']
    count = 1
    for i in range(len(ID_list)):
        if i<len(ID_list)-1:
            ID_data=testB_data.loc[:,ID_list[i]:ID_list[i+1]].drop(ID_list[i+1],axis=1)
            print ID_data
        else:
            ID_data=testB_data.loc[:,ID_list[i]:]
        ID_data.to_csv('C:/Users/user/Desktop/testB_tool_data/'+ID_list[i]+str(count)+'.csv',index=False)
        count+=1