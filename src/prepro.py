# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:57:42 2018

@author: varian
"""

from sklearn.model_selection import KFold

import pandas as pd
import os.path
import copy



from sklearn.model_selection import train_test_split

def split(data):
    if os.path.exists("datasets/R_train.csv"):
        Rtrain = pd.read_csv("datasets/R_train.csv")
        Rtest = pd.read_csv("datasets/R_test.csv")
    else:
        Rtrain, Rtest = train_test_split(data, test_size=0.2, shuffle=True)
        Rtrain.to_csv('datasets/R_train.csv', index=False)
        Rtest.to_csv('datasets/R_test.csv', index=False)
        split(data)
    return Rtrain, Rtest

def kFoldSplit(R, fold):
    my_dic = {}
    dic_of_dic = {}
    if os.path.exists("datasets/R_train"+str(fold-1)+".csv"):
        for i in range(fold):
            train = pd.read_csv("datasets/R_train"+str(i)+".csv")
            test = pd.read_csv("datasets/R_test"+str(i)+".csv")
            key_name_train = 'R_train'+str(i)
            key_name_test = 'R_test'+str(i)
            dic_of_dic[key_name_train] = key_name_train
            my_dic[key_name_train] = copy.deepcopy(train)
            dic_of_dic[key_name_test] = key_name_test
            my_dic[key_name_test] = copy.deepcopy(test)
    else:
        kf = KFold(n_splits=fold, random_state=1, shuffle=True)
        n=0
        for train_index, test_index in kf.split(R):
            R_train, R_test = R.iloc[train_index], R.iloc[test_index]
            R_train.to_csv('datasets/R_train'+str(n)+'.csv', index=False)
            R_test.to_csv('datasets/R_test'+str(n)+'.csv', index=False)
            n+=1
        kFoldSplit(R, fold)
    
    return my_dic, dic_of_dic