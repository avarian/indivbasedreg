# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 18:02:57 2018

@author: varian
"""

from src import prepro
from ast import literal_eval

import numpy as np
import pandas as pd
import math
import os.path
import sys

def loadR(path):
    if os.path.exists("datasets/"+path):
        data = pd.read_csv("datasets/"+path)
    else:
        prepro.split("subset_reviews.csv")
        data = loadR(path)
    return data

def loadUV(path):
    if os.path.exists("datasets/"+path):
        data = pd.read_csv("datasets/"+path, index_col=0)
    else:
        buildUV(path)
        data = loadUV(path)
    return data

def buildUV(path):
    k = 10
    if (path=="U.csv"):
        user = pd.read_csv("datasets/subset_users.csv")
        U = pd.DataFrame(np.random.rand(len(user), k), index=user.user_id, columns=range(k))
        U.to_csv('datasets/U.csv', index=True)
    if (path=="V.csv"):
        business = pd.read_csv("datasets/subset_business.csv")
        V = pd.DataFrame(np.random.rand(len(business), k), index=business.business_id, columns=range(k))
        V.to_csv('datasets/V.csv', index=True)
    return 0

def buildMatrixR(path, R, U, V):
    if os.path.exists("datasets/"+path+"UIMatrix.csv"):
        matrixR = pd.read_csv("datasets/"+path+"UIMatrix.csv", index_col=0)
    else:
        matrixR = pd.DataFrame(0, index=U.index, columns=V.index)
        for i in range(len(R)):
            matrixR[R.business_id[i]][R.user_id[i]] = R.stars[i]
        matrixR.to_csv('datasets/'+path+"UIMatrix.csv", index=True)
    return matrixR

def matStd(R_train_path, R_test_path, Rtrain, Rtest, U, V, alpha, lamb):
    list_index_train = indexNonZ(R_train_path, Rtrain)
    list_index_test = indexNonZ(R_test_path, Rtest)
    for x in range(len(list_index_train)):
        for y in range(len(list_index_train[x][1])):
            i = list_index_train[x][0]
            j = list_index_train[x][1][y]
            e = np.dot(U.loc[[i]],V.loc[[j]].T) - Rtrain.loc[i][j]
            U.loc[i] = U.loc[[i]].values - alpha * ( (e * V.loc[[j]].values) + (lamb * U.loc[[i]].values) )
            V.loc[j] = V.loc[[j]].values - alpha * ( (e * U.loc[[i]].values) + (lamb * V.loc[[j]].values) )
#            cost = e ** 2
#            U.loc[i], V.loc[j], cost = matStdupdate(Rtrain.loc[i][j], U.loc[[i]], V.loc[[j]], alpha, lamb)
            sys.stderr.write('\r~User: %d/%d~' % (x, len(list_index_train)))
            sys.stderr.flush()
    return U, V, list_index_train, list_index_test

def indivReg(R_train_path, R_test_path, Rtrain, Rtest, U, V, SN, alpha, lamb, beta):
    list_index_train = indexNonZ(R_train_path, Rtrain)
    list_index_test = indexNonZ(R_test_path, Rtest)
    SG = loadGrafSN(Rtrain, SN, list_index_train)
    index_SN = indexNonZ("socialNet", SG)
    for x in range(len(list_index_train)):
        for y in range(len(list_index_train[x][1])):
            i = list_index_train[x][0]
            j = list_index_train[x][1][y]
            e = np.dot(U.loc[[i]],V.loc[[j]].T) - Rtrain.loc[i][j]
            U.loc[i] = U.loc[[i]].values - alpha * ( (e * V.loc[[j]].values) + (lamb * U.loc[[i]].values) + (beta*sr_f(i,index_SN,U,SG)) )
            V.loc[j] = V.loc[[j]].values - alpha * ( (e * U.loc[[i]].values) + (lamb * V.loc[[j]].values) )
            sys.stderr.write('\r~User: %d/%d~' % (x, len(list_index_train)))
            sys.stderr.flush()
    
    return U, V, list_index_train, list_index_test

def loadMatIndx(path, R):
    if os.path.exists("datasets/"+path+"MatIndx.npy"):
        list_index=np.load("datasets/"+path+"MatIndx.npy")
    else:
        list_index = []
        for i in R.index:
            for j in R.columns:
                if (R.loc[i,j] > 0):
                    list_index.append(i+','+j)
        np.save("datasets/"+path+".npy", list_index)
    return list_index

def indexNonZ(path, R):
    if os.path.exists("datasets/"+path+"indexNonZ.npy"):
        index=np.load("datasets/"+path+"indexNonZ.npy")
    else:
        index=[]
        for i in R.index:
            temp1=[i]
            temp2=R.loc[i].nonzero()
            temp1.append(R.columns[temp2[0]])
            index.append(temp1)
        np.save("datasets/"+path+"indexNonZ.npy", index)
    return index

def matStdupdate(Rij, Ui, Vj, alpha, lamb):

    e = np.dot(Ui,Vj.T) - Rij

    u_temp = Ui.values - alpha * ( (e * Vj.values) + (lamb * Ui.values) )
    v_temp = Vj.values - alpha * ( (e * Ui.values) + (lamb * Vj.values) )

    cost = e ** 2

    return u_temp, v_temp, cost

def rmse(predictions, R, list_index):
    rmse = 0
    T = 0
    for x in range(len(list_index)):
        for y in range(len(list_index[x][1])):
            i = list_index[x][0]
            j = list_index[x][1][y]
            rmse += (predictions.loc[i][j]-R.loc[i][j])** 2
            T+=1
    return  np.sqrt(rmse/T)

def loadGrafSN(R, SN, index):
    if os.path.exists("datasets/socialNet.csv"):
        socialGraf = pd.read_csv("datasets/socialNet.csv", index_col=0)
    else:
        socialNet = pd.read_csv("datasets/"+SN, index_col=0)
        socialNet.loc[:,'list_friends']=socialNet.loc[:,'list_friends'].apply(lambda x: literal_eval(x))
        socialGraf = pd.DataFrame(0, index=socialNet.index, columns=socialNet.index)
    
        i=0
        for user in socialNet.index:
            j=0
            for friend in socialNet.list_friends[user]:
                x = R.loc[user]
                y = R.loc[friend]
                cor_pearson = pearson(x,y, user, index)
                socialGraf.loc[user,friend] = cor_pearson
                socialGraf.loc[friend,user] = cor_pearson
                sys.stderr.write('\r~~User: %d/%d  Friends: %d/%d~~~~~' % (i, len(socialNet), j, len(socialNet.list_friends[user])))
                sys.stderr.flush()
                j+=1
            i+=1
        socialGraf.to_csv('datasets/socialNet.csv', index=True)
    return socialGraf

def pearson(x, y, user, index):
    assert len(x) > 0
    assert len(x) == len(y)
    dense_X = []
    dense_Y = []    
    for i in range(len(index)):
        if index[i][0] == user:
            for j in range(len(index[i][1])):
                if (x[index[i][1][j]] > 0 and y[index[i][1][j]] > 0):
                    dense_X.append(x[index[i][1][j]])
                    dense_Y.append(y[index[i][1][j]]) 
            break
    n = len(dense_X)
    if n == 0:
        return 0
    avg_x = float(sum(dense_X)) / len(dense_X)
    avg_y = float(sum(dense_Y)) / len(dense_Y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = dense_X[idx] - avg_x
        ydiff = dense_Y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff
    sim = diffprod / math.sqrt(xdiff2 * ydiff2)
    if math.isnan(sim):
        sim = 0
    return sim

def sr_f(i,index_SN, U, SG):
    reg = 0
    for j in range(len(index_SN)):
        if (index_SN[j][0]==i):
            for k in range(len(index_SN[j][1])):
                f=index_SN[j][1][k]
                reg += SG.loc[i,f] * (U.loc[i].values - U.loc[f].values)
            return reg
#    print("WHAT????")
    return reg
