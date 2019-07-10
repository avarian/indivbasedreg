# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 17:59:22 2018

@author: varian
"""

from src import lorank
from src import prepro

import numpy as np
import pandas as pd
import sys
alpha = 0.01
lamb  = 0.01
beta  = 0.01
steps = 200

def low_rank(R, U, V):
    rmse_min = 100.0
    R_train, R_test = prepro.split(R)
    U1 = pd.DataFrame.copy(U);
    V1 = pd.DataFrame.copy(V);
    R_train_path = "R_train"
    R_trainM = lorank.buildMatrixR(R_train_path, R_train, U1, V1)
    R_test_path = "R_test"
    R_testM = lorank.buildMatrixR(R_test_path, R_test, U1, V1)
    rmse_train1 = []
    rmse_test1 = []
    for i in range(steps):
        sys.stderr.write('\r                   ~Step: %d/%d~' % (i+1, steps))
        U1temp, V1temp, list_index_train, list_index_test = lorank.matStd(R_train_path, R_test_path, R_trainM, R_testM, U1, V1, alpha, lamb)
        nR1 = U1.dot(V1.T)
        rmse_train1_temp = lorank.rmse(nR1, R_trainM, list_index_train)
        rmse_test1_temp = lorank.rmse(nR1, R_testM, list_index_test)
        if rmse_test1_temp < rmse_min:
            U1 = pd.DataFrame.copy(U1temp);
            V1 = pd.DataFrame.copy(V1temp);
            rmse_train1.append(rmse_train1_temp)
            rmse_test1.append(rmse_test1_temp)
            rmse_min = rmse_test1_temp
        else:
            break
        sys.stderr.flush()
    
    return U1, V1, rmse_train1, rmse_test1, nR1

def individual_based(R, U, V, SN):  
    rmse_min = 100.0
    R_train, R_test = prepro.split(R)
    U2 = pd.DataFrame.copy(U);
    V2 = pd.DataFrame.copy(V);
    R_train_path = "R_train"
    R_trainM = lorank.buildMatrixR(R_train_path, R_train, U2, V2)
    R_test_path = "R_test"
    R_testM = lorank.buildMatrixR(R_test_path, R_test, U2, V2)
    rmse_train2 = []
    rmse_test2 = []
    for i in range(steps):
        sys.stderr.write('\r                   ~Step: %d/%d~' % (i+1, steps))
        U2temp, V2temp, list_index_train, list_index_test = lorank.indivReg(R_train_path, R_test_path, R_trainM, R_testM, U2, V2, SN, alpha, lamb, beta)
        nR2 = U2.dot(V2.T)
        rmse_train2_temp = lorank.rmse(nR2, R_trainM, list_index_train)
        rmse_test2_temp = lorank.rmse(nR2, R_testM, list_index_test)
        if rmse_test2_temp < rmse_min:
            U2 = pd.DataFrame.copy(U2temp);
            V2 = pd.DataFrame.copy(V2temp);
            rmse_train2.append(rmse_train2_temp)
            rmse_test2.append(rmse_test2_temp)
            rmse_min = rmse_test2_temp
        else:
            break
        sys.stderr.flush()

    return U2, V2, rmse_train2, rmse_test2, nR2
if __name__ == "__main__":
    
    
#    Rtrain = lorank.loadR("Rtrain.csv")
#    Rtest = lorank.loadR("Rtest.csv")
    
    R = pd.read_csv("datasets/subset_reviews.csv")
    
    U = lorank.loadUV("U.csv")
    V = lorank.loadUV("V.csv")
    SN = "subset_users.csv"
    
#    RtrainUImat = lorank.buildMatrixR("RtrainUImat.csv", Rtrain, U, V)
#    RtestUImat  = lorank.buildMatrixR("RtestUImat.csv", Rtest, U, V)
#
    U1, V1, rmse_train1, rmse_test1, nR1 = low_rank(R, U, V)
#    np.save("result/rmse_train1.npy", rmse_train1)
#    np.save("result/rmse_test1.npy", rmse_test1)
#    V1.to_csv('result/V1.csv', index=True)
#    U1.to_csv('result/U1.csv', index=True)
#    
    U2, V2, rmse_train2, rmse_test2, nR2 = individual_based(R, U, V, SN)
#    np.save("result/rmse_train2.npy", rmse_train2)
#    np.save("result/rmse_test2.npy", rmse_test2)
#    V2.to_csv('result/V2.csv', index=True)
#    U2.to_csv('result/U2.csv', index=True)
#    
#    np.save("result/rmse_train1.npy", rmse_train1)
#    np.save("result/rmse_test1.npy", rmse_test1)
#    V1.to_csv('result/V1.csv', index=True)
#    U1.to_csv('result/U1.csv', index=True)