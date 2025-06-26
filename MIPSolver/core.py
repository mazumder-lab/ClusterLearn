import numpy as np
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB

import time

class MIPSolver:
    '''
    The Row Generation Exact Solver

    X, y: The training data in the dummified format
    lambda0, lambda1: Regularization coefficients for sparsity and clustering, respectively.
    groups: A list of what dummy variables belong to which categorical predictor.
    beta0: The warm-start
    M: The Big-M
    t_lim: Time limit
    iterations: Number of row generation iterations
    verbose: Whether to print the Gurobi log
    mip_tol: MIP gap tolerance
    '''
    
    def __init__(self, X, y, lambda0, lambda1, groups, beta0=None, M = 1,t_lim=1*60, iterations=5, verbose=1, mip_tol=0.005):
        self.X = deepcopy(np.array(X))
        self.p = (self.X.shape[1])
        self.n = (self.X.shape[0])
        self.y = np.array(y)
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.groups = groups
        self.r = len(groups)
        self.iterations = iterations
        self.verbose = verbose
        self.mip_tol = mip_tol

        
        self.n_max = -1
        for i in range(self.r):
            if len(self.groups[i])>self.n_max:
                self.n_max = len(self.groups[i])
        
        self.M = M
        
        if beta0 is None:
            self.beta = np.zeros(self.p)
            self.mu = 0
        else:
            self.beta = deepcopy(np.array(beta0[:-1]))
            self.mu = beta0[-1]

            idx = np.argwhere(self.beta > self.M)
            self.beta[idx] = self.M*np.ones(idx.shape)

            idx = np.argwhere(self.beta < -self.M)
            self.beta[idx] = -self.M*np.ones(idx.shape)
      


        
        

        
        self.t_lim = t_lim
        self.a = 2.1

        
    
    
    
        
    def GRB_rowgen(self):

        t0 = time.time()
        model = gp.Model('ClusterLearn')
        model.params.TimeLimit = self.t_lim
        model.params.LazyConstraints = 1
        model.params.OutputFlag = self.verbose
       

        z = model.addVars(self.p, lb = 0, ub = 1, vtype=GRB.BINARY, name="z")
        beta = model.addMVar(self.p,  lb=float('-inf'), name="beta")
        xi = model.addMVar(self.n,    lb=float('-inf'), name="xi")
        z_ijk = model.addMVar((self.n_max, self.n_max, self.r),lb = 0, ub = 1, vtype=GRB.BINARY, name="z_ijk")
        x = model.addVars(self.p, vtype=GRB.BINARY, name="x")
        mu = model.addMVar(1,name="mu", lb=float('-inf'))
        iterations = self.iterations

        beta.Start = deepcopy(self.beta)



        for j in range(self.r):
            g = np.array(self.groups[j])
            for i in range(self.n_max):
                for k in range(self.n_max):
                    if i>= len(g) or k >= len(g) or k > i:
                        model.addConstr(z_ijk[i,k,j] == 1)
                        continue
                    elif i == k:
                        model.addConstr(z_ijk[i,k,j] == 0)
            for i in range(len(g)):
                if i > 0:
                    model.addConstr(gp.quicksum(z_ijk[i,kk,j] for kk in range(i))  - x[g[i]] <= i-1)
            model.addConstr( x[g[0]] >= 1)
        
        
        model.addConstrs( beta[i] <= self.M*z[i] for i in range(self.p) )
        model.addConstrs( beta[i] >= - self.M*z[i] for i in range(self.p) )

        model.addConstr( xi - self.y + self.X@beta + mu*np.ones((self.n,1)) == 0)
        




        current_beta = deepcopy(self.beta)
        current_mu = deepcopy(self.mu)
        current_loc = np.where(np.abs(current_beta)>0)
        ub = self.objective(current_beta, current_mu)
        lb = - 1e10

        






        model.setObjective(  xi.T@xi/self.n  +
            self.lambda0*z.sum() + self.lambda1*x.sum(), GRB.MINIMIZE)
        
        for iter_c in range(iterations):
            for j in range(self.r):
                g = np.array(self.groups[j])
                if len(np.nonzero(current_beta[g])[0]) == 0:
                    continue
                for i in range(len(g)):
                    for k in range(i):
                            model.addConstr(beta[g[i]]-beta[g[k]]- self.a*self.M*z_ijk[i,k,j]<=0)
                            model.addConstr(beta[g[i]]-beta[g[k]]+ self.a*self.M*z_ijk[i,k,j]>=0)
        
            model.optimize()
            current_beta_new = deepcopy(beta.X)
            current_mu_new = deepcopy(mu.X)
            current_loc_new = np.where(np.abs(current_beta_new)>0)
            ub_new = self.objective(current_beta_new, current_mu_new)
            lb_new = model.ObjBound

            if ub_new < ub:
                ub = ub_new
                current_beta = deepcopy(current_beta_new)
                current_mu = deepcopy(current_mu_new)
                current_loc = deepcopy(current_loc_new)
            if lb_new > lb:
                lb = lb_new
            

            if np.array_equal(current_loc,current_loc_new) and model.MIPGap < self.mip_tol:
                break

            if time.time() - t0 > self.t_lim:
                break

        self.beta = deepcopy(current_beta)
        self.mu = deepcopy(current_mu)
        return self.beta, self.mu, self.objective(self.beta,self.mu), (ub-lb)/ub  
        
        





    def objective(self, beta, mu):
        r = self.y - self.X@beta - mu*np.ones(self.n)

        n_cluster = 0

        for j in range(self.r):
            g = np.array(self.groups[j])
            n_cluster += len(np.unique(beta[g]))

        nnz = len(np.nonzero(beta)[0])



        obj = np.linalg.norm(r)**2/self.n + self.lambda0*nnz + self.lambda1*n_cluster
        return obj
