
from utils import  performance_metrics,generate_random_correlated,BCD_wrapper
import numpy as np
from MIPSolver.core import MIPSolver






rng = 0
np.random.seed(rng)


n_cat = 5  # Number of categorical predictors
levels = 15 # Number of levels of each categorical predictor

### Create true beta^*
beta = [] 
for l in [[-2]*4 + [0]*(levels-8) + [2]*4 for j in range(2)]: # The first two categorical features get [-2,...,-2,0,...,0,2,...,2]
    beta += l
beta = beta + [0]*(n_cat*levels - 2*levels) # The rest of the categorical features get zero
###

### Draw the data
n = 500 # Number of train/test/validation data points 
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

### Run BCD for warm-start

lambda1 = 0.05
lambda0 = 0.05
beta_bcd, time_bcd = BCD_wrapper(X_cat_train, y, X_cat_val, y_val,numlevels_=[levels]*n_cat,l0_list=np.flip([lambda0]),l1_list=np.flip([lambda1]), l2_list=np.flip([0]))
   

### Calculate performance metrics for BCD
nnz_bcd, pred_bcd, _, purity_bcd, purity_nnz_bcd, nclusters_bcd = performance_metrics(beta_bcd[:-1], beta_star, groups, y_test, X_test,beta_bcd[-1], mu=np.mean(y))

### MIP Solver     
# Create the MIP solver
M = 1.2*np.max(np.abs(beta_bcd[:-1]))
mip_solver = MIPSolver( X, y, lambda0, lambda1, groups, beta0=beta_bcd, M = M)
# Solve the MIP
beta_mip, mu_mip, obj_mip, gap = mip_solver.GRB_rowgen()  

# Performance metrics for the exact solver      
nnz, pred, _, purity, purity_nnz, nclusters = performance_metrics(beta_mip, beta_star, groups, y_test, X_test,mu_mip, mu=np.mean(y))

# MIP solver results
print('MIP results:')
print("R2 on Test: ", pred, " Objective: ", obj_mip, " NNZ Purity: ", purity_nnz)

# BCD results  
print('BCD results:')    
print("R2 on Test: ", pred_bcd, " Objective: ", mip_solver.objective(beta_bcd[:-1],beta_bcd[-1]), " NNZ Purity: ", purity_nnz_bcd)
         
