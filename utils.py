
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import time
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
import ctypes
from ctypes import cdll
from sklearn.metrics import log_loss


def generate_random_correlated(n,cov_mat,num_levels,sparsity=0,clustering=0,RNG=0,noise_sigma=0,beta=None):
    '''
    The function to generate synthetic data

    beta can be used to pre-specify a vector of true coefficients. If a beta is provided, sparsity and clustering
    arguments are ignored.
    When no beta is specified, sparsity in [0,1] controls how sparse beta is : When 0,no sparsity is enforced and 
    when 1, beta is always sparse
    clustering in [0,1] controls how clustered the coefficeints in each group : When 0, each coefficient is 
    different. When 1, all levels have the same coffecient.
    n: number of data points to draw
    num_levels: number of levels for each categorical predictor
    noise_sigma: The standard deviation of the noise
    '''
    rpy2.robjects.numpy2ri.activate()
    pandas2ri.activate()
    importr('CatReg')
    r = robjects.r
    
    data = ro.conversion.get_conversion().rpy2py(r['CorrelatedDesignMatrix'](n,cov_mat,num_levels))
    data.columns = map(str,range(len(data.columns)))
    
    df = pd.DataFrame()

        
    for v in data.columns:
        diff_Vals = data[v].unique()
        dict_to_replace = {diff_Vals[i]:str(i) for i in range(len(diff_Vals)) }
        data[v].replace(dict_to_replace,inplace=True)
    
    for i in range(cov_mat.shape[0]):
        df[str(i)] = pd.Categorical(map(str, data[str(i)].astype(int)))
        
    data_dum = pd.get_dummies(df).sort_index(axis=1,key= lambda index: [(int(x.split('_')[0]),int(x.split('_')[1])) for x in index]  )
    p = len(data_dum.columns)
    
    beta_star = np.zeros(p)
    if not beta is None:
        beta_star = np.copy(beta)
    start_index = 0
    rng = np.random.RandomState(RNG)
    g_idx=[]
    num_non_zero_cats = int((1-sparsity)*p)
    for i in range(len(data.columns)):
        levels = data[data.columns[i]].unique()
        num_levels = len(levels)
        end_index = start_index + num_levels
        num_clusters = max(1,int((1-clustering)*(num_levels)))
        g_idx.append(list(range(start_index,end_index)))
        diff_values = rng.normal(0,10,num_clusters)
        if beta is None and i < num_non_zero_cats:
            beta_level = np.zeros(end_index - start_index)
            beta_level[rng.permutation(num_levels)[0:num_levels]] = list(np.concatenate((diff_values,rng.choice(diff_values,(num_levels - num_clusters),replace=True))))
            beta_star[start_index:end_index] = beta_level        
        start_index=end_index
        
    X = np.array(data_dum)
    X = X.astype(float)
    y = X@beta_star + rng.normal(0, noise_sigma, n)
    return np.array(data),X,y,beta_star,g_idx,df




def BCD_wrapper(X,y,Xval,yval,Xcont=None,Xcontval=None,classification=False,numlevels_=None,l0_list = [0.0],l1_list = [0.0],l2_list = [0.01],whole_block=False ):
    
    lib = cdll.LoadLibrary('../ClusterLearn/univariate/BCD_solver.so')
    lib.BCD_solve.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double) )
    lib.BCD_continuous_solve.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double) )
    
    lib.BCD_classfier_solve.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double) )
    lib.BCD_classfier_continuous_solve.argtypes = (ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_double) )

    n = len(X)
    nval = len(Xval)
    r = len(X[0])
    if not Xcont is None:
        rcont = len(Xcont[0])
    else:
        rcont = 0
    n_c = ctypes.c_int(n)
    nval_c = ctypes.c_int(nval)
    r_c = ctypes.c_int(r)
    rcont_c = ctypes.c_int(rcont)

    x = np.ascontiguousarray(X,dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    if classification:
        y = np.ascontiguousarray(y,dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        y = np.ascontiguousarray(y,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    xval = np.ascontiguousarray(Xval,dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    if classification:
        yval = np.ascontiguousarray(yval,dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        yval = np.ascontiguousarray(yval,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    if rcont >0:
        xcont = np.ascontiguousarray(Xcont,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        xcontval = np.ascontiguousarray(Xcontval,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    l0 = np.ascontiguousarray(l0_list,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    l1 = np.ascontiguousarray(l1_list,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    l2 = np.ascontiguousarray(l2_list,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    nl0_c = ctypes.c_int(len(l0_list))
    nl1_c = ctypes.c_int(len(l1_list))
    nl2_c = ctypes.c_int(len(l2_list))
    if numlevels_ is None:
        if rcont >0:
            numlevels = np.concatenate([(np.concatenate([X,Xval]).astype(np.int32).max(axis=0)+1), [rcont] ])
        else:
            numlevels = np.concatenate([X,Xval]).astype(np.int32).max(axis=0)+1
        numlevels = np.ascontiguousarray(numlevels,dtype=np.int32)
        # print(numlevels)
        p = numlevels.sum()
    else:
        numlevels = np.ascontiguousarray(numlevels_,dtype=np.int32)
        p = numlevels.sum()
    beta0 = np.zeros(p ,dtype=np.double).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    
    numlevels=numlevels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    beta = np.zeros(p+1,dtype=np.double)
    if rcont == 0:
        start = time.time()
        if classification :
            lib.BCD_classfier_solve(n_c,r_c,x,y,nval_c,xval,yval,l0,nl0_c,l1,nl1_c,l2,nl2_c,numlevels,beta0,beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        else:
            lib.BCD_solve(n_c,r_c,x,y,nval_c,xval,yval,l0,nl0_c,l1,nl1_c,l2,nl2_c,numlevels,beta0,beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        tend = time.time() - start
    else:
        start = time.time()
        if classification:
            lib.BCD_classfier_continuous_solve(n_c,r_c,rcont_c,x,xcont,y,nval_c,xval,xcontval,yval,l0,nl0_c,l1,nl1_c,l2,nl2_c,numlevels,beta0,beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        else:
            lib.BCD_continuous_solve(n_c,r_c,rcont_c,x,xcont,y,nval_c,xval,xcontval,yval,l0,nl0_c,l1,nl1_c,l2,nl2_c,numlevels,beta0,beta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        tend = time.time() - start
    return beta,tend




    acc = np.zeros((len(lambda1s), len(lambda2s)))
    best_accuracy = 0
    best_beta = 0
    best_intercept = 0
    start = time.time()
    for i in range(len(lambda1s)):
        for j in range(len(lambda2s)):
            lambda1 = lambda1s[i]
            lambda2 = lambda2s[j]
            if lambda1 == 0:
                regr =  LogisticRegression(penalty='l2',C=1/lambda2,random_state=0,solver='liblinear')
            elif lambda2 == 0:
                regr =  LogisticRegression(penalty='l1',C=1/lambda1,random_state=0,solver='liblinear')
            else:
                regr = LogisticRegression(penalty='elasticnet',C=1/(lambda1+2*lambda2),l1_ratio=lambda1/(lambda1+2*lambda2),random_state=0,solver='saga')
            regr.fit(X, y)
            #acc[i,j] = regr.score(Xval,yval)

            probabilty = 1/(np.exp(-Xval@regr.coef_.flatten() - regr.intercept_) + 1)
            logloss = log_loss(yval,probabilty,normalize=True)
            acc[i,j] = 1 - logloss

            if acc[i,j] > best_accuracy:
                best_accuracy = acc[i,j]
                best_beta = regr.coef_
                best_intercept = regr.intercept_
    end = time.time()
    return best_beta.flatten(),best_intercept[0], end - start


def classification_metrics(beta, groups, y_test, X_test,intercept=0):
    xtb = X_test@beta + intercept
    probabilty = 1/(np.exp(-xtb) + 1)
    logloss = log_loss(y_test,probabilty,normalize=True)
    accuracy =  (np.multiply(xtb,y_test) > 0).sum()/len(X_test)


    nnz = len(np.where(np.abs(beta)>0)[0])
    
    n_levels = 0
    for i in range(len(groups)):
        g = groups[i]
        g = np.array(g)
        g = g.astype(np.int32)
        n_levels = n_levels + len(np.unique(beta[g]))

    return nnz,accuracy,logloss,n_levels


def performance_metrics(beta, beta_star, groups, y_test, X_test,intercept=0,classification=False,only_pred=False, mu=0):
    
    if classification:
        estimation = np.nan
        xtb = X_test@beta + intercept
        probabilty = 1/(np.exp(-xtb) + 1)
        logloss = log_loss(y_test,probabilty,normalize=True)
        accuracy =  (np.multiply(xtb,y_test) > 0).sum()/len(X_test)
        res = accuracy
    else:
        estimation = np.linalg.norm(beta-beta_star)/np.linalg.norm(beta_star)
        prediction = 1 - (np.linalg.norm((y_test-X_test@beta) - intercept)/np.linalg.norm(y_test -np.ones(len(y_test))*mu))**2
        res = prediction
    

    nnz = len(np.where(np.abs(beta)>0)[0])
    
    n_levels = 0
    for i in range(len(groups)):
        g = groups[i]
        g = np.array(g)
        g = g.astype(np.int32)
        n_levels = n_levels + len(np.unique(beta[g]))
    
    if only_pred:
        return nnz, res, n_levels

    purity = 0
    purity_nnz = 0
    nnz_counter = 0
    for r in range(len(groups)):
        g = np.array(groups[r])

        if np.linalg.norm(beta_star[g]) > 1e-10:
                nnz_counter += len(g)


        true_cluster = np.zeros(len(g))
        assigned_cluster = np.zeros(len(g))
        true_uniq = np.unique(beta_star[g])
        est_uniq = np.unique(beta[g])
        for i in range(len(g)):
            true_cluster[i] = np.where(true_uniq == beta_star[g[i]])[0][0]
            assigned_cluster[i] = np.where(est_uniq == beta[g[i]])[0][0]
         
        m = np.zeros((len(true_uniq),len(est_uniq))) 
        

        for i in range(len(true_uniq)):
            for j in range(len(est_uniq)):
                idx1 = np.where(true_cluster == i)[0]
                idx2 = np.where(assigned_cluster == j)[0]
                m[i,j] = len(np.intersect1d(idx1,idx2))
                     
        for j in range(len(est_uniq)):
            purity = purity + np.max(m[:,j])
            if np.linalg.norm(beta_star[g]) > 1e-10:
                purity_nnz = purity_nnz + np.max(m[:,j])
               
    purity = purity/len(beta)
    try:
        purity_nnz /= nnz_counter
    except:
        purity_nnz = -1
    



    if classification:
        return nnz, res, estimation, purity, n_levels,logloss
    
    return nnz, res, estimation, purity, purity_nnz, n_levels