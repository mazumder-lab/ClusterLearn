import pandas as pd
import numpy as np
import sys 
import os 
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

sys.path.append('..')

from ClusterLearn.utils import validate_logistic_scope, performance_metrics, BCD_wrapper

counter_sigma = 1
df_data = []

acc_cd = []
acc_cd0 = []
acc_enet = []
acc_scope = []


levels_cd = []
levels_cd0 = []
levels_enet = []
levels_scope = []

time_cd = []
time_cd0 = []
time_enet = []
time_scope = []



for counter_run in range(5):



    df = pd.read_csv('./insurance/train.csv')
    df = df.reset_index(drop=True)
    y = (df['Response'] > 4).apply(lambda x : 2*int(x) - 1).reset_index(drop=True)
    
    n = len(df)
    pos_y = y[y>0].index
    neg_y = y[y<=0].index
    print(len(pos_y),len(neg_y))
    n = 15200*2

    obs_to_use = list(np.random.choice(pos_y,n//2, replace=False)) + list(np.random.choice(neg_y,n//2, replace=False))
    df = df.iloc[obs_to_use]
    df = df.reset_index(drop=True)


    continuous_variables = '''Product_Info_4, Ins_Age, Ht, Wt, BMI, Employment_Info_1, Employment_Info_4, Employment_Info_6, Insurance_History_5, Family_Hist_2, Family_Hist_3, Family_Hist_4, Family_Hist_5'''.split(", ")
    categorical_variables = '''Product_Info_1, Product_Info_2, Product_Info_3, Product_Info_5, Product_Info_6, Product_Info_7, Employment_Info_2, Employment_Info_3, Employment_Info_5, InsuredInfo_1, InsuredInfo_2, InsuredInfo_3, InsuredInfo_4, InsuredInfo_5, InsuredInfo_6, InsuredInfo_7, Insurance_History_1, Insurance_History_2, Insurance_History_3, Insurance_History_4, Insurance_History_7, Insurance_History_8, Insurance_History_9, Family_Hist_1, Medical_History_2, Medical_History_3, Medical_History_4, Medical_History_5, Medical_History_6, Medical_History_7, Medical_History_8, Medical_History_9, Medical_History_11, Medical_History_12, Medical_History_13, Medical_History_14, Medical_History_16, Medical_History_17, Medical_History_18, Medical_History_19, Medical_History_20, Medical_History_21, Medical_History_22, Medical_History_23, Medical_History_25, Medical_History_26, Medical_History_27, Medical_History_28, Medical_History_29, Medical_History_30, Medical_History_31, Medical_History_33, Medical_History_34, Medical_History_35, Medical_History_36, Medical_History_37, Medical_History_38, Medical_History_39, Medical_History_40, Medical_History_41'''.split(", ")

    df = df[categorical_variables+continuous_variables  + ['Response']]

    na_threshold = 0.8
    df = df.dropna(axis=1,thresh=na_threshold*len(df))

    na = df.isna().any()
    for c in na[na].index:
        df[c] = df[c].fillna(df[c].mean())
    
    df = df.reset_index(drop=True)

    ntrain = 1000

    train_val = np.random.choice(range(n),3*ntrain, replace=False)


    training_index = train_val[:ntrain]
    validation_index = train_val[ntrain:3*ntrain]
    test_index = [i for i in range(n) if (not i in training_index) and (not i in validation_index)]

    traininig_levels = {c:df.iloc[training_index][c].unique() for c in categorical_variables}


    validation_index = [i for i in validation_index if np.array([df.loc[i,c] in traininig_levels[c]  for c in categorical_variables]).all() ]
    test_index = [i for i in test_index if np.array([df.loc[i,c] in traininig_levels[c]  for c in categorical_variables]).all() ]

    print(len(training_index),len(validation_index),len(test_index))


    df = df.loc[list(training_index) + list(validation_index) + list(test_index)]
    y = (df['Response'] > 4).apply(lambda x : 2*int(x) - 1).reset_index(drop=True)
    b_y = (df['Response'] > 4).reset_index(drop=True).astype(int)

    df.drop(columns=['Response'],inplace=True)

    training_index = list(range(len(training_index)))
    validation_index = list(range(len(training_index), len(training_index) + len(validation_index)))
    test_index = list(range(len(training_index) + len(validation_index),len(training_index) + len(validation_index) + len(test_index)))
    df = df.reset_index(drop=True)
    print(b_y[training_index].mean(),b_y[validation_index].mean(),b_y[test_index].mean())

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
    X_cont = np.ascontiguousarray(X_cont,dtype=np.double)
    print('p is ',X.shape[1])

    groups = []
    start_index = 0;
    for c in df_cat.columns:
        groups.append(list(range(start_index,start_index + df_cat[c].nunique())))
        start_index += df_cat[c].nunique()

    np.random.seed(None)


    
    n_lambda = 20
    lambda1cd = np.logspace(-5,2,n_lambda**2)
    lambda1l0 = np.logspace(-5,2,n_lambda)
    lambda0l0 = np.logspace(-5,2,n_lambda)

    print('counter_run ----> ',counter_run)
    data = []

    beta_bcd, time_bcd = BCD_wrapper(X_cat_non_dummy[training_index], y[training_index], X_cat_non_dummy[validation_index], y[validation_index],X_cont[training_index],X_cont[validation_index],classification=True,numlevels_=levels,l0_list=np.flip(['0']),l2_list=np.flip(['0']),l1_list=np.flip(lambda1cd),whole_block=False)
    data.append({'method':'CD','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})
    
    
    beta_bcd, time_bcd = BCD_wrapper(X_cat_non_dummy[training_index], y[training_index], X_cat_non_dummy[validation_index], y[validation_index],X_cont[training_index],X_cont[validation_index],classification=True,numlevels_=levels,l0_list=np.flip(lambda0l0),l1_list=np.flip(lambda1l0),whole_block=False)
    data.append({'method':'CD+L0','beta':beta_bcd[:-1],'time':time_bcd,'intercept':beta_bcd[-1]})

    # beta_scope, t_scope = validate_logistic_scope(df.iloc[training_index], df.iloc[test_index], b_y[training_index], len(X[0]),len(df_cat.columns),rcont=len(df_cont.columns),gamma=100,nfolds=5,nlambda=25)
    # data.append({'method':'SCOPE','beta':beta_scope[:-1],'time':t_scope,'intercept':beta_scope[-1]})

    for method in data:
        method['run'] = counter_run
        nnz, acc,nlevels= performance_metrics(method['beta'], np.zeros_like(method['beta']), groups, y[test_index], X[test_index],method['intercept'],classification=True,only_pred=True)
        if 'pred' in method:
            acc = ((method['pred'][:,0] > 0.5) == (b_y[test_index])).mean()
        method.update({'nnz':nnz, 'acc': acc, 'n':n,'nlevels':nlevels})


        if method['method'] == 'CD':
            acc_cd.append(acc)
            levels_cd.append(nlevels)
            time_cd.append(method['time'])
        elif method['method'] == 'CD+L0':
            acc_cd0.append(acc)
            levels_cd0.append(nlevels)
            time_cd0.append(method['time'])
        elif method['method'] == 'SCOPE':
            acc_scope.append(acc)
            levels_scope.append(nlevels)
            time_scope.append(method['time'])



data = []
n_run = len(acc_cd)

data.append({'method':'CD+L0', 'Accuracy' : np.mean(acc_cd0), 'Accuracy_std': np.std(acc_cd0)/np.sqrt(n_run), 'levels' : np.mean(levels_cd0), 'levelsstd': np.std(levels_cd0)/np.sqrt(n_run), 'time' : np.mean(time_cd0), 'timestd': np.std(time_cd0)/np.sqrt(n_run)})
data.append({'method':'CD', 'Accuracy' : np.mean(acc_cd), 'Accuracy_std': np.std(acc_cd)/np.sqrt(n_run), 'levels' : np.mean(levels_cd), 'levelsstd': np.std(levels_cd)/np.sqrt(n_run), 'time' : np.mean(time_cd), 'timestd': np.std(time_cd)/np.sqrt(n_run)})
data.append({'method':'SCOPE', 'Accuracy' : np.mean(acc_scope), 'Accurqcy_std': np.std(acc_scope)/np.sqrt(n_run), 'levels' : np.mean(levels_scope), 'levelsstd': np.std(levels_scope)/np.sqrt(n_run), 'time' : np.mean(time_scope), 'timestd': np.std(time_scope)/np.sqrt(n_run)})


df = pd.DataFrame(data)
print(df)
