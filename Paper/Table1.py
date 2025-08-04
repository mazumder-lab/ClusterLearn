import pandas as pd
import numpy as np
import sys 
import os 
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

sys.path.append('..')

from ClusterLearn.utils import validate_scope, performance_metrics, BCD_wrapper, validate_Elasticnet

counter_sigma = 1
df_data = []

error_cd = []
error_cd0 = []
error_enet = []
error_scope = []


levels_cd = []
levels_cd0 = []
levels_enet = []
levels_scope = []

time_cd = []
time_cd0 = []
time_enet = []
time_scope = []



for counter_run in range(100):



    df = pd.read_csv('./bike/hour.csv')
    df = df.reset_index(drop=True)
    y = (df['cnt']).reset_index(drop=True)

    n = len(df)
    print(n)


    continuous_variables = ['atemp','temp',  'hum', 'windspeed']
    categorical_variables = [c for c in df.columns if not c in continuous_variables and c != 'cnt'  and c != 'dteday' and c != 'instant'  and c != 'casual'  and c != 'registered' ]



    df = df[categorical_variables+continuous_variables  + ['cnt']]
    df = df.reset_index(drop=True)

    ntrain = 100
    train_val = np.random.choice(range(n),2*ntrain, replace=False)

    training_index = train_val[:ntrain]
    validation_index = train_val[ntrain:2*ntrain]
    test_index = [i for i in range(n) if (not i in training_index) and (not i in validation_index)]

    traininig_levels = {c:df.iloc[training_index][c].unique() for c in categorical_variables}


    validation_index = [i for i in validation_index if np.array([df.loc[i,c] in traininig_levels[c]  for c in categorical_variables]).all() ]
    test_index = [i for i in test_index if np.array([df.loc[i,c] in traininig_levels[c]  for c in categorical_variables]).all() ]

    print(len(training_index),len(validation_index),len(test_index))


    df = df.loc[list(training_index) + list(validation_index) + list(test_index)]
    y = (df['cnt']).reset_index(drop=True)

    df.drop(columns=['cnt'],inplace=True)


    training_index = list(range(len(training_index)))
    validation_index = list(range(len(training_index), len(training_index) + len(validation_index)))
    test_index = list(range(len(training_index) + len(validation_index),len(training_index) + len(validation_index) + len(test_index)))
    df = df.reset_index(drop=True)
    
    columns = list(df.columns)

    ##Replace columns in categorical variables with indexes
    df.columns = map(str,range(len(df.columns)))


    ##Replace levels in categorical varialbes with indexes
    for v in df.columns:
        if columns[int(v)] in categorical_variables:
            diff_vals = df[v].unique()
            dict_to_replace = {diff_vals[i]:str(i) for i in range(len(diff_vals)) }
            df[v].replace(dict_to_replace,inplace=True)
            df[v] = pd.Categorical(map(str, df[v].astype(int)))
        else:
            df[v] = df[v].astype(float)

    df_cont = df[[c for c in df.columns if columns[int(c)] in continuous_variables]]
    df_cat = df[[c for c in df.columns if columns[int(c)] in categorical_variables]]


    df_cat_dummy = pd.get_dummies(df_cat).sort_index(axis=1,key= lambda index: [(int(x.split('_')[0]),int(x.split('_')[1])) for x in index]  )
    levels = [df_cat[c].nunique() for c in df_cat.columns] + [len(df_cont.columns)]


    X_cont = np.array(df_cont)
    X_cat = np.array(df_cat_dummy)
    n = len(df)
    X = np.concatenate([X_cat,X_cont],axis=1)
    X_cat_non_dummy = np.array(df_cat)
    X_cat_non_dummy = np.ascontiguousarray(X_cat_non_dummy,dtype=np.int32)
    print('p is ',X.shape[1])


    groups = []
    start_index = 0;
    for c in df_cat.columns:
        groups.append(list(range(start_index,start_index + df_cat[c].nunique())))
        start_index += df_cat[c].nunique()

    np.random.seed(counter_run)



    n_lambda = 20
    lambda1cd = np.logspace(-5,2,n_lambda**2)
    lambda1l0 = np.logspace(-5,2,n_lambda)
    lambda0l0 = np.logspace(-5,2,n_lambda)

    print('counter_run ----> ',counter_run)
    data = []

    beta_bcd, time_bcd = BCD_wrapper(X_cat_non_dummy[training_index], y[training_index], X_cat_non_dummy[validation_index], y[validation_index],X_cont[training_index],X_cont[validation_index],classification=False,numlevels_=levels,l0_list=np.flip(['0']),l2_list=np.flip(['0']),l1_list=np.flip(lambda1cd),whole_block=False)
    data.append({'method':'CD','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})
    
    beta_bcd, time_bcd = BCD_wrapper(X_cat_non_dummy[training_index], y[training_index], X_cat_non_dummy[validation_index], y[validation_index],X_cont[training_index],X_cont[validation_index],classification=False,numlevels_=levels,l0_list=np.flip(lambda0l0),l1_list=np.flip(lambda1l0),whole_block=False)
    data.append({'method':'CD+L0','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})

    beta_scope, t_scope = validate_scope(df.iloc[training_index], X[validation_index], y[training_index], y[validation_index], len(X[0]),len(df_cat.columns),rcont=len(df_cont.columns),nfolds=1,n_lambda=n_lambda**2, lambda_min=1e-5, lambda_max=100)
    data.append({'method':'SCOPE','beta':beta_scope[:-1],'time':t_scope,'intercept':beta_scope[-1]})
    

    beta_elastic, intercept_elastic,time = validate_Elasticnet(X[training_index],y[training_index],X[validation_index],y[validation_index],lambda0l0,lambda0l0)
    data.append({'method':'Elastic-net','beta':beta_elastic,'time':time,'intercept':intercept_elastic})

    df_data= []

    for method in data:
        method['run'] = counter_run
        nnz, res, n_levels = performance_metrics(method['beta'], np.zeros_like(method['beta']), groups, y[test_index], X[test_index],method['intercept'],classification=False,only_pred=True, mu=np.mean(y[training_index]))
        beta = method['beta'][groups[-1]]
        method.update({'nnz':nnz, 'res': res, 'nlevels':n_levels})
        df_data.append(method)

        if method['method'] == 'CD':
            error_cd.append(res)
            levels_cd.append(n_levels)
            time_cd.append(method['time'])
        elif method['method'] == 'CD+L0':
            error_cd0.append(res)
            levels_cd0.append(n_levels)
            time_cd0.append(method['time'])
        elif method['method'] == 'SCOPE':
            error_scope.append(res)
            levels_scope.append(n_levels)
            time_scope.append(method['time'])
        elif method['method'] == 'Elastic-net':
            error_enet.append(res)
            levels_enet.append(n_levels)
            time_enet.append(method['time'])



data = []
n_run = len(error_cd)

data.append({'method':'CD+L0', 'R2' : np.mean(error_cd0), 'R2std': np.std(error_cd0)/np.sqrt(n_run), 'levels' : np.mean(levels_cd0), 'levelsstd': np.std(levels_cd0)/np.sqrt(n_run), 'time' : np.mean(time_cd0), 'timestd': np.std(time_cd0)/np.sqrt(n_run)})
data.append({'method':'CD', 'R2' : np.mean(error_cd), 'R2std': np.std(error_cd)/np.sqrt(n_run), 'levels' : np.mean(levels_cd), 'levelsstd': np.std(levels_cd)/np.sqrt(n_run), 'time' : np.mean(time_cd), 'timestd': np.std(time_cd)/np.sqrt(n_run)})
data.append({'method':'SCOPE', 'R2' : np.mean(error_scope), 'R2std': np.std(error_scope)/np.sqrt(n_run), 'levels' : np.mean(levels_scope), 'levelsstd': np.std(levels_scope)/np.sqrt(n_run), 'time' : np.mean(time_scope), 'timestd': np.std(time_scope)/np.sqrt(n_run)})
data.append({'method':'Elastic-Net', 'R2' : np.mean(error_enet), 'R2std': np.std(error_enet)/np.sqrt(n_run), 'levels' : np.mean(levels_enet), 'levelsstd': np.std(levels_enet)/np.sqrt(n_run), 'time' : np.mean(time_enet), 'timestd': np.std(time_enet)/np.sqrt(n_run)})


df = pd.DataFrame(data)
print(df)
