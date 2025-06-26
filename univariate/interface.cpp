//YourFile.cpp (compiled into a .dll or .so file)
#include "Solver.h"
#include <vector>
#include <stdlib.h> 
#include <iostream>
#include <memory>
#include <fstream>

/*
extern "C" void prox_dp(int n, double *y, double l0,double l1,double l2, double *beta) {
    
    std::vector<double> y_vec(n);
    for(int i = 0; i < n;i++) {
        //std::cout << "here   " <<i << std::endl;
        y_vec[i] = y[i];
        }
    auto solver = SegSolver(y_vec,l0,l1,l2);
    std::vector<double> beta_vec = solver.backtrace();
    for(int i = 0; i < n; i++){
        beta[i] = beta_vec[i];
    }
}
extern "C" void prox_dp_w(int n, double *y,double *w, double l0,double l1,double l2, double *beta) {
    
    std::vector<double> y_vec(n);
    std::vector<double> w_vec(n);
    for(int i = 0; i < n;i++) {
        //std::cout << "here   " <<i << std::endl;
        y_vec[i] = y[i];
        w_vec[i] = w[i];
        }
    auto solver = SegSolver(y_vec,w_vec,l0,l1,l2);
    std::vector<double> beta_vec = solver.backtrace();
    for(int i = 0; i < n; i++){
        beta[i] = beta_vec[i];
    }
}
*/

extern "C" void BCD_solve(int n,int r, int *x, double *y,int nval,int *xval, double *yval, double* l0,int nl0,double* l1,int nl1,double* l2,int nl2, int* num_levels,double * beta0 ,double *beta) {
    auto y_vec = std::make_shared<std::vector<double> >(y,y+n);
    auto x_vec = std::make_shared< std::vector<std::vector<int> > > (n, std::vector<int>(r,-1));

    auto yval_vec = std::vector<double> (yval,yval+nval);
    auto xval_vec = std::vector<std::vector<int> >  (nval, std::vector<int>(r,0));

    auto beta0_vec = std::make_shared< std::vector<std::vector<double> > > (r);

    auto l0_vec = std::vector<double>  (l0, l0+nl0);
    auto l1_vec = std::vector<double>  (l1, l1+nl1);
    auto l2_vec = std::vector<double> (l2, l2+nl2);

    for(int i = 0; i < n;i++)
     {
        for(int j = 0; j < r;j++){
            x_vec->at(i)[j] = x[i*r + j];
        }
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < r;j++){
            xval_vec.at(i)[j] = xval[i*r + j];
        }
    }
    int agg_num_levels = 0;
    for(int j = 0; j < r;j++){
        beta0_vec->at(j) = std::vector<double>(beta0+agg_num_levels,beta0 + agg_num_levels + num_levels[j]);
        agg_num_levels += num_levels[j];
        //for(auto x : beta0_vec->at(j)){
        //   std::cout << x << " " << std::flush;
        //}
        //std::cout << std::endl;
    }
    auto solver = new Solver(x_vec,y_vec,l0_vec,l1_vec,l2_vec,beta0_vec);
    std::vector<std::vector<double> >  beta_vec = solver->find_best_beta(xval_vec,yval_vec);
    int flat_index = 0;
    for(int i = 0; i <= r; i++){
        for(int j = 0; j < beta_vec[i].size(); j++){
            beta[flat_index] = beta_vec[i][j];
            flat_index++;
        }
    }
}

extern "C" void BCD_classfier_solve(int n,int r, int *x, int *y,int nval,int *xval, int *yval, double* l0,int nl0,double* l1,int nl1,double* l2,int nl2, int* num_levels,double * beta0 ,double *beta) {
    auto y_vec = std::make_shared<std::vector<int> >(y,y+n);
    auto x_vec = std::make_shared< std::vector<std::vector<int> > > (n, std::vector<int>(r,-1));

    auto yval_vec = std::vector<int> (yval,yval+nval);
    auto xval_vec = std::vector<std::vector<int> >  (nval, std::vector<int>(r,0));

    auto beta0_vec = std::make_shared< std::vector<std::vector<double> > > (r);

    auto l0_vec = std::vector<double>  (l0, l0+nl0);
    auto l1_vec = std::vector<double>  (l1, l1+nl1);
    auto l2_vec = std::vector<double> (l2, l2+nl2);

    
    for(int i = 0; i < n;i++)
     {
        for(int j = 0; j < r;j++){
            x_vec->at(i)[j] = x[i*r + j];
        }
        
        
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < r;j++){
            xval_vec.at(i)[j] = xval[i*r + j];
        }
    }
    int agg_num_levels = 0;
    for(int j = 0; j < r;j++){
        beta0_vec->at(j) = std::vector<double>(beta0+agg_num_levels,beta0 + agg_num_levels + num_levels[j]);
        agg_num_levels += num_levels[j];
    }
    auto solver = new Solver(x_vec,y_vec,l0_vec,l1_vec,l2_vec,beta0_vec);
    std::vector<std::vector<double> >  beta_vec = solver->find_best_beta(xval_vec,yval_vec,"log_loss");
    int flat_index = 0;
    for(int i = 0; i <= r; i++){
        for(int j = 0; j < beta_vec[i].size(); j++){
            beta[flat_index] = beta_vec[i][j];
            flat_index++;
        }
    }
}


extern "C" void BCD_continuous_solve(int n,int r,int rcont, int *x,double* xcont, double *y,int nval,int *xval, double * xcontval, double *yval, double* l0,int nl0,double* l1,int nl1,double* l2,int nl2, int* num_levels,double * beta0 ,double *beta) {

    

    auto y_vec = std::make_shared<std::vector<double> >(y,y+n);
    auto x_vec = std::make_shared< std::vector<std::vector<int> > > (n, std::vector<int>(r,-1));
    auto xcont_vec = std::make_shared< std::vector<std::vector<double> > > (n, std::vector<double>(rcont,-1));

    auto yval_vec = std::vector<double> (yval,yval+nval);
    auto xval_vec = std::vector<std::vector<int> >  (nval, std::vector<int>(r,0));
    auto xcontval_vec = std::vector<std::vector<double> >  (nval, std::vector<double>(rcont,0));

    auto beta0_vec = std::make_shared< std::vector<std::vector<double> > > (r+1);

    auto l0_vec = std::vector<double>  (l0, l0+nl0);
    auto l1_vec = std::vector<double>  (l1, l1+nl1);
    auto l2_vec = std::vector<double> (l2, l2+nl2);

    for(int i = 0; i < n;i++)
     {
        for(int j = 0; j < r;j++){
            x_vec->at(i)[j] = x[i*r + j];
        }
        
        
    }
    
    for(int i = 0; i < n;i++)
     { 
        for(int j = 0; j < rcont;j++){
            xcont_vec->at(i)[j] = xcont[i*rcont + j];

        }
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < r;j++){
            xval_vec.at(i)[j] = xval[i*r + j];
        }
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < rcont;j++){
            xcontval_vec.at(i)[j] = xcontval[i*rcont + j];
        }
    }
    // Close the file
    int agg_num_levels = 0;
    //num_levels[r] = rcont
    for(int j = 0; j < r+1;j++){
        beta0_vec->at(j) = std::vector<double>(beta0+agg_num_levels,beta0 + agg_num_levels + num_levels[j]);
        agg_num_levels += num_levels[j];
        
    }
    auto solver = new Solver(x_vec,xcont_vec,y_vec,l0_vec,l1_vec,l2_vec,beta0_vec);
    std::vector<std::vector<double> >  beta_vec = solver->find_best_beta(xval_vec,xcontval_vec,yval_vec);
    int flat_index = 0;
    for(int i = 0; i <= r+1; i++){
        for(int j = 0; j < beta_vec[i].size(); j++){
            beta[flat_index] = beta_vec[i][j];
            flat_index++;
        }
    }
}

extern "C" void BCD_classfier_continuous_solve(int n,int r,int rcont, int *x,double* xcont, int *y,int nval,int *xval, double * xcontval, int *yval, double* l0,int nl0,double* l1,int nl1,double* l2,int nl2, int* num_levels,double * beta0 ,double *beta) {

    std::ofstream log_file;
    log_file.open("interface.log");
    log_file << "Starting" << std::flush << std::endl;

    auto y_vec = std::make_shared<std::vector<int> >(y,y+n);
    auto x_vec = std::make_shared< std::vector<std::vector<int> > > (n, std::vector<int>(r,-1));
    auto xcont_vec = std::make_shared< std::vector<std::vector<double> > > (n, std::vector<double>(rcont,-1));

    auto yval_vec = std::vector<int> (yval,yval+nval);
    auto xval_vec = std::vector<std::vector<int> >  (nval, std::vector<int>(r,0));
    auto xcontval_vec = std::vector<std::vector<double> >  (nval, std::vector<double>(rcont,0));

    auto beta0_vec = std::make_shared< std::vector<std::vector<double> > > (r+1);

    auto l0_vec = std::vector<double>  (l0, l0+nl0);
    auto l1_vec = std::vector<double>  (l1, l1+nl1);
    auto l2_vec = std::vector<double> (l2, l2+nl2);

    for(int i = 0; i < n;i++)
     {
        for(int j = 0; j < r;j++){
            x_vec->at(i)[j] = x[i*r + j];
        }
        
        
    }
    
    for(int i = 0; i < n;i++)
     { 
        for(int j = 0; j < rcont;j++){
            xcont_vec->at(i)[j] = xcont[i*rcont + j];

        }
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < r;j++){
            xval_vec.at(i)[j] = xval[i*r + j];
        }
    }
    for(int i = 0; i < nval;i++)
     {
        for(int j = 0; j < rcont;j++){
            xcontval_vec.at(i)[j] = xcontval[i*rcont + j];
        }
    }
    // Close the file
    int agg_num_levels = 0;
    //num_levels[r] = rcont
    for(int j = 0; j < r+1;j++){
        beta0_vec->at(j) = std::vector<double>(beta0+agg_num_levels,beta0 + agg_num_levels + num_levels[j]);
        agg_num_levels += num_levels[j];
        
    }
    log_file << "Calling Solver" << std::flush << std::endl;
    auto solver = new Solver(x_vec,xcont_vec,y_vec,l0_vec,l1_vec,l2_vec,beta0_vec);
    log_file << "Fiding best beta" << std::flush << std::endl;
    std::vector<std::vector<double> >  beta_vec = solver->find_best_beta(xval_vec,xcontval_vec,yval_vec,"log_loss");
    int flat_index = 0;
    for(int i = 0; i <= r+1; i++){
        for(int j = 0; j < beta_vec[i].size(); j++){
            beta[flat_index] = beta_vec[i][j];
            flat_index++;
        }
    }
}