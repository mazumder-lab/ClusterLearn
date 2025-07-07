import pandas as pd
import numpy as np
import sys 
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=RuntimeWarning)

sys.path.append('..')

from ClusterLearn.utils import BCD_wrapper

counter_sigma = 1
df_data = []

np.random.seed(97)

lambda1 = [0.5,1,75]
lambda0 = [50,0]


df = pd.read_csv('./bike/hour.csv')
df = df.reset_index(drop=True)
y = (df['cnt']).reset_index(drop=True)

n = len(df)
print(n)


continuous_variables = []
categorical_variables = ['hr','weekday']




df = df[categorical_variables+continuous_variables  + ['cnt']]


df = df.reset_index(drop=True)

ntrain =100
train_val = np.random.choice(range(n),2*ntrain, replace=False)

training_index = train_val[:ntrain]
traininig_levels = {c:df.iloc[training_index][c].unique() for c in categorical_variables}



df = df.loc[list(training_index)]
y = (df['cnt']).reset_index(drop=True)

df.drop(columns=['cnt'],inplace=True)


training_index = list(range(len(training_index)))
df = df.reset_index(drop=True)
columns = list(df.columns)

# ##Replace columns in categorical variables with indexes
df.columns = map(str,range(len(df.columns)))


##Replace levels in categorical varialbes with indexes
for v in df.columns:
    if columns[int(v)] in categorical_variables:
        diff_vals = np.sort(df[v].unique())
        print(diff_vals)
        print(len(diff_vals))
        dict_to_replace = {diff_vals[i]:str(i) for i in range(len(diff_vals)) }
        print(dict_to_replace)
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
print(X.shape)
X_cat_non_dummy = np.array(df_cat)
X_cat_non_dummy = np.ascontiguousarray(X_cat_non_dummy,dtype=np.int32)
print('p is ',X.shape[1])

data = []


clusterlearn_levels = np.zeros(len(lambda1))
clusterlearn_nz = np.zeros(len(lambda1))
clusterlearn_r2 = np.zeros(len(lambda1))
clusterlearn_l1 = np.zeros(len(lambda1))


clusterlearnl0_levels = np.zeros(len(lambda1)*len(lambda0))
clusterlearnl0_r2 = np.zeros(len(lambda1)*len(lambda0))
clusterlearnl0_l1 = np.zeros(len(lambda1)*len(lambda0))
clusterlearnl0_l0 = np.zeros(len(lambda1)*len(lambda0))
counter_1 = 0
y = y.astype(float)

y[training_index] = y[training_index] + .0*np.random.normal(len(training_index))

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=17) 
matplotlib.rc('ytick', labelsize=17) 



print('---------')
idx = None
l2 = 0
iii = 0
marker = ['rv','b*']
fmts = ['b-','r--']
counter_l0 = 0
l2 = 1e-6
for l1 in (lambda1):
    ax = plt.gca()
    counter_1 = 0
    iii = 0
    idx1 = None
    idx2 = None
    for l0 in lambda0:
        beta_bcd, time_bcd = BCD_wrapper(X_cat_non_dummy[training_index], y[training_index],X_cat_non_dummy[training_index], y[training_index],None,None,classification=False,numlevels_=None,l0_list=np.flip([l0]),l2_list=np.flip([l2]),l1_list=np.flip([l1]),whole_block=False)
        beta = beta_bcd[:-1]
        alpha = beta_bcd[-1]

        if idx1 is None:
            idx1 = np.argsort(beta[:23])
            print(beta[:23][idx1])
            idx2 = np.argsort(beta[23:])
            print(idx1)
            print(idx2)

        counter_1 += 1
        beta_sorted =  np.zeros(len(beta))
        beta_sorted[:23] = beta[:23][idx1]
        beta_sorted[23:] = beta[23:][idx2]
        plt.plot(beta_sorted,marker[iii],markersize=10)
        iii += 1

    plt.legend(['$\lambda_0$ = '+str(l0) for l0 in lambda0],loc='upper left',prop={'size': 15})
    plt.show()




