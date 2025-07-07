
import os.path
import sys
sys.path.append('..')
from ClusterLearn.utils import validate_scope, performance_metrics, generate_random_correlated, BCD_wrapper, validate_Elasticnet
import numpy as np
import time

import os
import matplotlib.pyplot as plt
os.makedirs("../ClusterLearn/results", exist_ok=True)
import pandas as pd
import time

PREFIX = "../ClusterLearn/results/results_fig_3"

rng = 0

N_run = 50

df_data = []



N = [100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

noise_sigma = 2
rho = 0.2

n_cat=20
levels = 30
beta = []
n_non_l = 5
for l in [[-2]*10 + [0]*(levels-20) + [2]*10 for j in range(n_non_l)]:
    beta += l
beta = beta + [0]*(n_cat*levels - levels*n_non_l)
cov_mat = (1-rho)*np.eye(n_cat) + rho*np.ones((n_cat,n_cat))

    
for counter_run in range(N_run):    
    for n in N:
        print(n,rho,counter_run)

        FILE = "_" + str(n) +  "_" + str(noise_sigma) + "_" + str(rho) + "_" + str(counter_run) + ".csv"
        FILE = PREFIX + FILE
        if os.path.isfile(FILE):
            continue
        data = []
        rng = rng + 1

        X_cat0, X0, y0, beta_star, groups, df=generate_random_correlated(3*n,cov_mat,levels,RNG=rng,noise_sigma=noise_sigma,sparsity=0,clustering=0,beta=list(map(float,beta)))
        X = X0[0:n,:]
        df_train = df.iloc[0:n]
        X_val = X0[n:2*n,:]
        X_test = X0[2*n:3*n,:]
        y = y0[0:n]
        y_val = y0[n:2*n]
        y_test = y0[2*n:3*n]

        X_cat0_train = X_cat0[0:n]
        X_cat0_val = X_cat0[n:2*n,:]
        X_cat0_test = X_cat0[2*n:3*n,:]






        n = len(X)
        p = len(X[0])

        n_lambda_per = 15

        t_scope = 0
        beta_scope = np.zeros(p)
        beta_scope, t_scope = validate_scope(df_train, X_val, y, y_val, p,n_cat, n_lambda_per**2, lambda_min=1e-5, lambda_max=10,gamma=8 )
        data.append({'method':'SCOPE-8','beta':beta_scope[:-1],'time':t_scope,'intercept':beta_scope[-1]})
    


        lambda1 = np.logspace(-5,1,n_lambda_per**2)
        lambda0 = np.logspace(-5,1,n_lambda_per)
        lambda2 = np.logspace(-5,1,n_lambda_per)


        beta_elastic, intercept_elastic , time_elastic = validate_Elasticnet(X,y,X_val,y_val,lambda0,lambda2)
        data.append({'method':'ELASTIC_NET','beta':beta_elastic,'time':time_elastic,'intercept':intercept_elastic})

        beta_bcd, time_bcd = BCD_wrapper(X_cat0_train, y, X_cat0_val, y_val,numlevels_=[levels]*n_cat,l0_list=[0],l1_list=np.flip(lambda1))
        data.append({'method':'CD','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})

        beta_bcd, time_bcd = BCD_wrapper(X_cat0_train, y, X_cat0_val, y_val,numlevels_=[levels]*n_cat,l0_list=np.flip(lambda0),l1_list=np.flip(lambda2))
        data.append({'method':'CD+L0','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})

        # print(data)
        for method in data:
            method['run'] = counter_run
            method['sigma'] = noise_sigma
            method['rho'] = rho
            method['sparsity'] = 0
            method['clustering'] = 0
            nnz, pred, est, purity, purity_nnz, nlevels= performance_metrics(method['beta'], beta_star, groups, y_test, X_test,method['intercept'], mu=np.mean(y))
            method.update({'nnz':nnz, 'pred': pred, 'est': est, 'purity':purity,'n':n,'ncat':n_cat,'nlevels':nlevels, 'purity_nnz': purity_nnz})
            print(n,rho,counter_run,method['method'],method['pred'], method['purity_nnz'], method['nlevels'])
            df_data.append(method)

        dff=pd.DataFrame([{k:i[k] for k in i if k != 'beta'} for i in data])

        dff.to_csv(FILE,index=False,)


pred_cdl0 = np.zeros(len(N))
pred_cd = np.zeros(len(N))
pred_scope = np.zeros(len(N))
pred_enet = np.zeros(len(N))

pred_cdl0_error = np.zeros(len(N))
pred_cd_error = np.zeros(len(N))
pred_scope_error = np.zeros(len(N))
pred_enet_error = np.zeros(len(N))


purity_cdl0 = np.zeros(len(N))
purity_cd = np.zeros(len(N))
purity_scope = np.zeros(len(N))
purity_enet = np.zeros(len(N))


purity_cdl0_error= np.zeros(len(N))
purity_cd_error = np.zeros(len(N))
purity_scope_error = np.zeros(len(N))
purity_enet_error = np.zeros(len(N))

levels_cdl0 = np.zeros(len(N))
levels_cd = np.zeros(len(N))
levels_scope = np.zeros(len(N))
levels_enet = np.zeros(len(N))


levels_cdl0_error = np.zeros(len(N))
levels_cd_error = np.zeros(len(N))
levels_scope_error = np.zeros(len(N))
levels_enet_error = np.zeros(len(N))

counter_n = 0
for n in N:
    pred_cdl0_n = np.zeros(N_run)
    pred_cd_n = np.zeros(N_run)
    pred_scope_n = np.zeros(N_run)
    pred_enet_n = np.zeros(N_run)

    purity_cdl0_n = np.zeros(N_run)
    purity_cd_n = np.zeros(N_run)
    purity_scope_n = np.zeros(N_run)
    purity_enet_n = np.zeros(N_run)


    levels_cdl0_n = np.zeros(N_run)
    levels_cd_n = np.zeros(N_run)
    levels_scope_n = np.zeros(N_run)
    levels_enet_n = np.zeros(N_run)




    for run in range(N_run):
        FILE = "_" + str(n) +  "_" + str(noise_sigma) + "_" + str(rho) + "_" + str(run) + ".csv"
        FILE = PREFIX + FILE
        df = pd.read_csv(FILE)


        df0 = df.loc[df['method']=='CD+L0']
        pred_cdl0_n[run] = float(df0['pred'])
        purity_cdl0_n[run] = float(df0['purity_nnz'])
        levels_cdl0_n[run] = float(df0['nlevels'])



        

        df0 = df.loc[df['method']=='CD']
        pred_cd_n[run] = float(df0['pred'])
        purity_cd_n[run] = float(df0['purity_nnz'])
        levels_cd_n[run] = float(df0['nlevels'])



        df0 = df.loc[df['method']=='SCOPE-8']
        pred_scope_n[run] = float(df0['pred'])
        purity_scope_n[run] = float(df0['purity_nnz'])
        levels_scope_n[run] = float(df0['nlevels'])


        

        df0 = df.loc[df['method']=='ELASTIC_NET']
        pred_enet_n[run] = float(df0['pred'])
        purity_enet_n[run] = float(df0['purity_nnz'])
        levels_enet_n[run] = float(df0['nlevels'])





    pred_cdl0[counter_n] = np.mean(pred_cdl0_n)
    pred_cd[counter_n] = np.mean(pred_cd_n)
    pred_scope[counter_n] = np.mean(pred_scope_n)
    pred_enet[counter_n] = np.mean(pred_enet_n)

    


    pred_cdl0_error[counter_n] = np.std(pred_cdl0_n)/np.sqrt(N_run)
    pred_cd_error[counter_n] = np.std(pred_cd_n)/np.sqrt(N_run)
    pred_scope_error[counter_n] = np.std(pred_scope_n)/np.sqrt(N_run)
    pred_enet_error[counter_n] = np.std(pred_enet_n)/np.sqrt(N_run)



    purity_cdl0[counter_n] = np.mean(purity_cdl0_n)
    purity_cd[counter_n] = np.mean(purity_cd_n)
    purity_scope[counter_n] = np.mean(purity_scope_n)
    purity_enet[counter_n] = np.mean(purity_enet_n)


    purity_cdl0_error[counter_n] = np.std(purity_cdl0_n)/np.sqrt(N_run)
    purity_cd_error[counter_n] = np.std(purity_cd_n)/np.sqrt(N_run)
    purity_scope_error[counter_n] = np.std(purity_scope_n)/np.sqrt(N_run)
    purity_enet_error[counter_n] = np.std(purity_enet_n)/np.sqrt(N_run)





    levels_cdl0[counter_n] = np.mean(levels_cdl0_n)
    levels_cd[counter_n] = np.mean(levels_cd_n)
    levels_scope[counter_n] = np.mean(levels_scope_n)
    levels_enet[counter_n] = np.mean(levels_enet_n)


    levels_cdl0_error[counter_n] = np.std(levels_cdl0_n)/np.sqrt(N_run)
    levels_cd_error[counter_n] = np.std(levels_cd_n)/np.sqrt(N_run)
    levels_scope_error[counter_n] = np.std(levels_scope_n)/np.sqrt(N_run)
    levels_enet_error[counter_n] = np.std(levels_enet_n)/np.sqrt(N_run)



    counter_n +=1

import matplotlib
matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 

plt.errorbar(N,pred_cdl0,yerr =pred_cdl0_error,fmt='ok-' ,linewidth=2.5 )
plt.errorbar(N,pred_cd,yerr =pred_cd_error,fmt= '^r-',linewidth=2.5)
plt.errorbar(N,pred_enet,yerr =pred_enet_error,fmt= '2g--',linewidth=2.5)
plt.errorbar(N,pred_scope,yerr =pred_scope_error,fmt= 'sb--',linewidth=2.5)
plt.xscale('log')
plt.legend(['ClusterLearn-L0','ClusterLearn','Elastic Net', 'SCOPE'],ncol=2, loc='lower right',prop={'size': 12})
plt.show()

plt.errorbar(N,purity_cdl0,yerr =purity_cdl0_error,fmt='ok-'  )
plt.errorbar(N,purity_cd,yerr =purity_cd_error,fmt= '^r-')
plt.errorbar(N,purity_scope,yerr =purity_scope_error,fmt= 'sb--')
plt.xscale('log')
plt.legend(['ClusterLearn-L0','ClusterLearn','SCOPE'],ncol=2)

plt.show()

matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 

plt.errorbar(N,levels_cdl0,yerr =levels_cdl0_error,fmt='ok-' ,linewidth=2.5 )
plt.errorbar(N,levels_cd,yerr =levels_cd_error,fmt= '^r-',linewidth=2.5)
plt.errorbar(N,levels_scope,yerr =levels_scope_error,fmt= 'sb--',linewidth=2.5)
plt.xscale('log')
plt.legend(['ClusterLearn-L0','ClusterLearn','SCOPE'],ncol=2, loc='right',prop={'size': 12})

plt.show()