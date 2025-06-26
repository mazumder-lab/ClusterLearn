
from utils import  performance_metrics,generate_random_correlated,BCD_wrapper
import numpy as np






rng = 0
np.random.seed(rng)


n_cat=10  # Number of categorical predictors
levels = 12 # Number of levels of each categorical predictor

### Create true beta^*
beta = [] 
for l in [[-2]*4 + [0]*(levels-8) + [2]*4 for j in range(2)]: # The first two categorical features get [-2,...,-2,0,...,0,2,...,2]
    beta += l
beta = beta + [0]*(n_cat*levels - 2*levels) # The rest of the categorical features get zero
###

### Draw the data
n = 100 # Number of train/test/validation data points 
noise_sigma = 1 # Noise standard deviation
rho = 0.2 # Correlation between categorical predictors
cov_mat = (1-rho)*np.eye(n_cat) + rho*np.ones((n_cat,n_cat)) # Pairwise correlation of rho
X_cat0, X0, y0, beta_star, groups,_ = generate_random_correlated(3*n,cov_mat,levels,RNG=rng,noise_sigma=noise_sigma,sparsity=0,clustering=0,beta=list(map(float,beta))) # Draw the data
###


### Split the data
# Dummified:
X = X0[0:n,:] # Train
X_val = X0[n:2*n,:] # Validation
X_test = X0[2*n:3*n,:] # Test
# Factors:
X_cat_train = X_cat0[0:n] # Train
X_cat_val = X_cat0[n:2*n,:] # Validation
X_cat_test = X_cat0[2*n:3*n,:] # Test

y = y0[0:n]
y_val = y0[n:2*n]
y_test = y0[2*n:3*n]
###

### Tuning parameter grid
n_lambda = 10
lambda1 = np.logspace(-2,2,n_lambda)
lambda0 = np.logspace(-2,2,n_lambda)


### Run ClusterLearn
beta_bcd, time_bcd = BCD_wrapper(X_cat_train, y, X_cat_val, y_val,numlevels_=[levels]*n_cat,l0_list=np.flip(lambda0),l1_list=np.flip(lambda1))
   
# Results   
intercept = beta_bcd[-1]
beta_bcd = beta_bcd[:-1]


### Calculate performance metrics
nnz, pred, _, purity, purity_nnz, nclusters= performance_metrics(beta_bcd, beta_star, groups, y_test, X_test,intercept, mu=np.mean(y))
print("R2 on Test: ", pred, " Number of Regression Coefficient Clusters: ", nclusters, " NNZ Purity: ", purity_nnz)
                
     
                
