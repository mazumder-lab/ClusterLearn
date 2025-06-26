#include "Solver.h"
#include <math.h>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <string>

Solver::Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<double> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_) {
    VERBOSE = verbose_;
    if (VERBOSE){
        log_file.open("output" + std::to_string(std::time(0)) + ".log");
    }
    X = X_;
    y = y_;
    lambda0 = l0;
    lambda1 = l1;
    lambda2 = l2;
    n = X->size();
    r = X->at(0).size();
    beta = std::vector<std::vector<std::vector< std::vector<std::vector<double> > > > >(lambda0.size());
    beta0 = beta0_;
    for(int i = 0; i < lambda0.size(); i++){
        beta[i] = std::vector<std::vector< std::vector<std::vector<double> > >  >(lambda1.size());
        for(int j = 0; j < lambda1.size();j++){
            beta[i][j] = std::vector< std::vector<std::vector<double> > >  (lambda2.size());
            for(int k = 0; k < lambda2.size();k++){
                beta[i][j][k] =  std::vector<std::vector<double> > (r+1); //last vector holds intercept
                for(int l = 0; l < r; l++){
                    beta[i][j][k][l] = beta0->at(l);
                }
                beta[i][j][k][r] = std::vector<double>(1,0); //intercept
            }
        }
    }
}

Solver::Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<int> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_) {
    VERBOSE = verbose_;
    if (VERBOSE){
        log_file.open("output" + std::to_string(std::time(0)) + ".log");
    }
    X = X_;
    y_cat = y_;
    lambda0 = l0;
    lambda1 = l1;
    lambda2 = l2;
    n = X->size();
    r = X->at(0).size();
    beta = std::vector<std::vector<std::vector< std::vector<std::vector<double> > > > >(lambda0.size());
    beta0 = beta0_;
    for(int i = 0; i < lambda0.size(); i++){
        beta[i] = std::vector<std::vector< std::vector<std::vector<double> > >  >(lambda1.size());
        for(int j = 0; j < lambda1.size();j++){
            beta[i][j] = std::vector< std::vector<std::vector<double> > >  (lambda2.size());
            for(int k = 0; k < lambda2.size();k++){
                beta[i][j][k] =  std::vector<std::vector<double> > (r+1); //last vector holds intercept
                for(int l = 0; l < r; l++){
                    beta[i][j][k][l] = beta0->at(l);
                }
                beta[i][j][k][r] = std::vector<double>(1,0); //intercept
            }
        }
    }
}

Solver::Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<std::vector<double> > > & X_cont_ ,std::shared_ptr<std::vector<double> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_) {
    VERBOSE = verbose_;
    if (VERBOSE){
        log_file.open("output" + std::to_string(std::time(0)) + ".log");
    }
    X = X_;
    X_cont = X_cont_;
    y = y_;
    lambda0 = l0;
    lambda1 = l1;
    lambda2 = l2;
    n = X->size();
    r = X->at(0).size();

    X_cont_m = Eigen::MatrixXd(n,X_cont->at(0).size());
    for(int i = 0; i < n;i++){
        X_cont_m.row(i) = Eigen::VectorXd::Map(&(X_cont->at(i)[0]),X_cont->at(i).size());
    }
    

    beta = std::vector<std::vector<std::vector< std::vector<std::vector<double> > > > >(lambda0.size());
    beta0 = beta0_;
    for(int i = 0; i < lambda0.size(); i++){
        beta[i] = std::vector<std::vector< std::vector<std::vector<double> > >  >(lambda1.size());
        for(int j = 0; j < lambda1.size();j++){
            beta[i][j] = std::vector< std::vector<std::vector<double> > >  (lambda2.size());
            for(int k = 0; k < lambda2.size();k++){
                beta[i][j][k] =  std::vector<std::vector<double> > (r+2); //last vector holds intercept, the one before last holds the continuous betas
                for(int l = 0; l <= r; l++){
                    beta[i][j][k][l] = beta0->at(l);
                }
                beta[i][j][k][r+1] = std::vector<double>(1,0); //intercept
            }
        }
    }
}

Solver::Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<std::vector<double> > > & X_cont_ ,std::shared_ptr<std::vector<int> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_) {
    VERBOSE = verbose_;
    if (VERBOSE){
        log_file.open("output" + std::to_string(std::time(0)) + ".log");
    }
    X = X_;
    X_cont = X_cont_;
    y_cat = y_;
    lambda0 = l0;
    lambda1 = l1;
    lambda2 = l2;
    n = X->size();
    r = X->at(0).size();

    if(VERBOSE){
        log_file << "Creating solver " << std::flush << std::endl;
    }

    X_cont_m = Eigen::MatrixXd(n,X_cont->at(0).size());
    for(int i = 0; i < n;i++){
        X_cont_m.row(i) = Eigen::VectorXd::Map(&(X_cont->at(i)[0]),X_cont->at(i).size());
    }
    
    if(VERBOSE){
        log_file << "Init beta " << std::flush << std::endl;
    }

    beta = std::vector<std::vector<std::vector< std::vector<std::vector<double> > > > >(lambda0.size());
    beta0 = beta0_;
    for(int i = 0; i < lambda0.size(); i++){
        beta[i] = std::vector<std::vector< std::vector<std::vector<double> > >  >(lambda1.size());
        for(int j = 0; j < lambda1.size();j++){
            beta[i][j] = std::vector< std::vector<std::vector<double> > >  (lambda2.size());
            for(int k = 0; k < lambda2.size();k++){
                beta[i][j][k] =  std::vector<std::vector<double> > (r+2); //last vector holds intercept, the one before last holds the continuous betas
                for(int l = 0; l <= r; l++){
                    beta[i][j][k][l] = beta0->at(l);
                }
                beta[i][j][k][r+1] = std::vector<double>(1,0); //intercept
            }
        }
    }
    if(VERBOSE){
        log_file << "Done Creating solver " << std::flush << std::endl;
    }
}

double Solver::BCD(int i_l0, int i_l1, int i_l2){
    int MAXITER = 50;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }

    double diff_old_new = INFINITY;
    int current_iter = 0;

    auto old_beta = beta[i_l0][i_l1][i_l2];

    double obj=0;


    Eigen::VectorXd residuals = Eigen::VectorXd::Zero(n);

    for(int i =0; i < n; i++){
        residuals[i] = y->at(i);
        for(int j = 0; j < r; j++){
            residuals[i] -= old_beta[j][X->at(i)[j]];
        }
        if(!(X_cont == nullptr)){
            for(int j = 0; j < old_beta[r].size();j++){
                residuals[i] -= old_beta[r][j]*X_cont->at(i)[j];
            }
        }
        //remove intercept
        residuals[i] -= old_beta[intercept_index][0];
        obj += residuals[i]*residuals[i];
    }
    obj /= n;
    if(VERBOSE){
            log_file << "obj before pnealty is " << obj << std::flush << std::endl;
    }
    obj += penalty(i_l0,  i_l1,  i_l2);
    if(VERBOSE){
            log_file << "Starting BCD with l0,l1,l2= " << lambda0[i_l0] << " , " << lambda1[i_l1] << " , " << lambda2[i_l2] << " with penalty " << penalty(i_l0,  i_l1,  i_l2) << std::flush << std::endl;
    }
    while(current_iter <= MAXITER && diff_old_new > TOL){
        if(VERBOSE){
            log_file << "Iteration " << current_iter << " Objective : " << obj << std::flush << std::endl;
        }
        for(int i = 0; i < r; i++){
            std::vector<double> residual_averages_tmp(old_beta[i].size(),0.0);
            std::vector<std::pair<double,int> > residual_averages(old_beta[i].size());
            std::vector<int> residual_averages_count(old_beta[i].size(),0);
            for(int j = 0; j <n;j++){
                 //remove current beta's contribution
                residuals[j] += beta[i_l0][i_l1][i_l2][i][X->at(j)[i]];
                residual_averages_tmp[X->at(j)[i]] += residuals[j];
                residual_averages_count[X->at(j)[i]] += 1;
            }
            for(int j = 0; j < residual_averages.size();j++){
                if (residual_averages_count[j] == 0){
                    residual_averages[j] = std::make_pair<>(0.0, j);
                }
                else{
                    residual_averages[j] = std::make_pair<>(residual_averages_tmp[j]/((double)residual_averages_count[j]), j);
                }
                
            }
            std::sort(residual_averages.begin(),residual_averages.end(), std::greater<>());
            
            std::vector<double> sorted_y,sorted_w;
            for (int j = 0; j < residual_averages.size(); j++)
            {
                sorted_y.push_back(residual_averages[j].first);
                sorted_w.push_back(residual_averages_count[residual_averages[j].second]);
                
            }
            
            SegSolver block_solver = SegSolver(sorted_y,sorted_w, lambda0[i_l0]*(n/2.0),lambda1[i_l1]*(n/2.0),lambda2[i_l2]*(n/2.0));
            std::vector<double> new_beta_sorted = block_solver.backtrace();
            
            for (int j = 0; j < residual_averages.size(); j++)
            {
                beta[i_l0][i_l1][i_l2][i][residual_averages[j].second]= new_beta_sorted[j];
            }

            //Add current beta contribution to residuals
            for(int j = 0; j <n;j++){
                 //add current beta's contribution
                residuals[j] -= beta[i_l0][i_l1][i_l2][i][X->at(j)[i]];
            }
            
        } //End of loop over groups

        /////update continuous variables
        if(!(X_cont == nullptr)){
            for(int i = 0; i <n;i++){
                //remove current beta's contribution
                for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size();j++){
                    residuals[i] += beta[i_l0][i_l1][i_l2][r][j]* X_cont->at(i)[j];
                }
            }

            //compute and update new beta
            Eigen::VectorXd beta_new;
            if (lambda2[i_l2] == 0) {
                Eigen::BDCSVD<Eigen::MatrixXd> svd(X_cont_m,Eigen::ComputeThinU | Eigen::ComputeThinV);
                beta_new = svd.solve(residuals);
            }
            else {
                Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X_cont_m.cols(), X_cont_m.cols());
                beta_new = (X_cont_m.transpose()*X_cont_m + n*lambda2[i_l2]*eye).ldlt().solve(X_cont_m.transpose()*residuals);
            }
            for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size();j++){
                    beta[i_l0][i_l1][i_l2][r][j] = beta_new[j];
            }
            for(int i = 0; i <n;i++){
                //remove current beta's contribution
                for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size();j++){
                    residuals[i] -= beta[i_l0][i_l1][i_l2][r][j]* X_cont->at(i)[j];
                }
            }


        }

        ///////////Update intercept
        double mean_residuals = 0;
        for(int j = 0; j <n;j++){
            //remove current beta's contribution
            residuals[j] += beta[i_l0][i_l1][i_l2][intercept_index][0];
            mean_residuals += residuals[j];

        }
        beta[i_l0][i_l1][i_l2][intercept_index][0] = mean_residuals/n;
        obj = 0;
        for(int j = 0; j <n;j++){
            //add current beta's contribution
            residuals[j] -= beta[i_l0][i_l1][i_l2][intercept_index][0];
            obj += residuals[j]*residuals[j];
        }
        obj /= n;
        obj += penalty(i_l0,  i_l1,  i_l2);
        ////////////////////////////

        //////See how much progress we made
        diff_old_new = 0;
        double beta_old_norm = 0;
        for(int i = 0; i < old_beta.size()-1; i++){
            for(int j = 0; j < old_beta[i].size(); j++){
                diff_old_new += (old_beta[i][j] - beta[i_l0][i_l1][i_l2][i][j])*(old_beta[i][j] - beta[i_l0][i_l1][i_l2][i][j]);
                beta_old_norm += old_beta[i][j]*old_beta[i][j];
            }
        }
        diff_old_new = diff_old_new/beta_old_norm;
        old_beta = beta[i_l0][i_l1][i_l2];
        current_iter++;

    }
    return obj;
}

double Solver::BCD_sigmoid_prox(int i_l0, int i_l1, int i_l2){
    int MAXITER = 50;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }

    double diff_old_new = INFINITY;
    int current_iter = 0;

    auto old_beta = beta[i_l0][i_l1][i_l2];

    double obj=0;


    Eigen::VectorXd xtb = Eigen::VectorXd::Zero(n);

    for(int i =0; i < n; i++){
        for(int j = 0; j < r; j++){
            xtb[i] += old_beta[j][X->at(i)[j]];
        }
        if(!(X_cont == nullptr)){
            for(int j = 0; j < old_beta[r].size();j++){
                xtb[i] += old_beta[r][j]*X_cont->at(i)[j];
            }
        }
        //add intercept
        xtb[i] += old_beta[intercept_index][0];
        obj += std::log(1 + std::exp(-y_cat->at(i)*xtb[i]));
    }
    obj /= n;
    obj += penalty(i_l0,  i_l1,  i_l2);
    if(VERBOSE){
            log_file << "Starting BCD with l0,l1,l2= " << lambda0[i_l0] << " , " << lambda1[i_l1] << " , " << lambda2[i_l2] << " with penalty " << penalty(i_l0,  i_l1,  i_l2) << std::flush << std::endl;
    }
    while(current_iter <= MAXITER && diff_old_new > TOL){
        if(VERBOSE){
            log_file << "Iteration " << current_iter << " Objective : " << obj << std::flush << std::endl;
        }
        for(int i = 0; i < r; i++){
            std::vector<double> residual_averages_tmp(old_beta[i].size(),0.0);
            std::vector<std::pair<double,int> > residual_averages(old_beta[i].size());
            std::vector<int> residual_averages_count(old_beta[i].size(),0);
            for(int j = 0; j <n;j++){
                 //remove current beta's contribution
                residual_averages_tmp[X->at(j)[i]] += old_beta[i][X->at(j)[i]] + (2.0* y_cat->at(j) / (1.0+ std::exp(y_cat->at(j) * xtb[j]))   );
                residual_averages_count[X->at(j)[i]] += 1;
                
            }
            for(int j = 0; j < residual_averages.size();j++){
                if (residual_averages_count[j] == 0){
                    residual_averages[j] = std::make_pair<>(0.0, j);
                }
                else{
                    residual_averages[j] = std::make_pair<>(residual_averages_tmp[j]/((double)residual_averages_count[j]), j);
                }
                
                
            }
            std::sort(residual_averages.begin(),residual_averages.end(), std::greater<>());
            
            std::vector<double> sorted_y,sorted_w;
            for (int j = 0; j < residual_averages.size(); j++)
            {
                sorted_y.push_back(residual_averages[j].first);
                sorted_w.push_back(residual_averages_count[residual_averages[j].second]/4.0);
                
            }
            
            SegSolver block_solver = SegSolver(sorted_y,sorted_w, lambda0[i_l0]*(n/2.0),lambda1[i_l1]*(n/2.0),lambda2[i_l2]*(n/2.0));
            std::vector<double> new_beta_sorted = block_solver.backtrace();
            
            for (int j = 0; j < residual_averages.size(); j++)
            {
                beta[i_l0][i_l1][i_l2][i][residual_averages[j].second]= new_beta_sorted[j];
            }

            //Add current beta contribution to residuals
            for(int j = 0; j <n;j++){
                 //add current beta's contribution
                xtb[j] += beta[i_l0][i_l1][i_l2][i][X->at(j)[i]] - old_beta[i][X->at(j)[i]] ;
            }
            
        } //End of loop over groups



        /////update continuous variables
        if(!(X_cont == nullptr)){
            Eigen::VectorXd y_tilde = Eigen::VectorXd::Zero(n);
            for(int i = 0; i <n;i++){
                //remove current beta's contribution
                double xtb_cont = 0;
                for(int j = 0; j < old_beta[r].size();j++){
                    xtb_cont += beta[i_l0][i_l1][i_l2][r][j]*X_cont->at(i)[j];
                }
                y_tilde[i] = xtb_cont + 2.0*y_cat->at(i)/(1.0 + std::exp(y_cat->at(i)*xtb[i]));
            }

            //compute and update new beta
            Eigen::VectorXd beta_new;
            if (lambda2[i_l2]== 0) {
                Eigen::BDCSVD<Eigen::MatrixXd> svd(X_cont_m,Eigen::ComputeThinU | Eigen::ComputeThinV);
                beta_new = svd.solve(y_tilde);
            }
            else {
                Eigen::MatrixXd eye = Eigen::MatrixXd::Identity(X_cont_m.cols(), X_cont_m.cols());
                beta_new = (X_cont_m.transpose()*X_cont_m + n*lambda2[i_l2]*eye).ldlt().solve(X_cont_m.transpose()*y_tilde);
            }
            for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size();j++){
                    beta[i_l0][i_l1][i_l2][r][j] = beta_new[j];
            }
            for(int i = 0; i <n;i++){
                for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size();j++){
                        xtb[i] += beta[i_l0][i_l1][i_l2][r][j]*X_cont->at(i)[j]  - old_beta[r][j]*X_cont->at(i)[j];
                }
            }
        }
        
        ///////////Update intercept
        double mean_residuals = 0;
        for(int j = 0; j <n;j++){
            //remove current beta's contribution
            mean_residuals += beta[i_l0][i_l1][i_l2][intercept_index][0]  + 2.0*y_cat->at(j)/(1.0 + std::exp(y_cat->at(j)*xtb[j]));

        }
        beta[i_l0][i_l1][i_l2][intercept_index][0] = mean_residuals/n;
        obj = 0;
        for(int j = 0; j <n;j++){
            //add current beta's contribution
            xtb[j] += beta[i_l0][i_l1][i_l2][intercept_index][0] - old_beta[intercept_index][0];
            obj += std::log(1 + std::exp(-y_cat->at(j)*xtb[j]));
        }
        obj /= n;
        obj += penalty(i_l0,  i_l1,  i_l2);
        ////////////////////////////


       

        //////See how much progress we made
        diff_old_new = 0;
        double beta_old_norm = 0;
        for(int i = 0; i < old_beta.size()-1; i++){
            for(int j = 0; j < old_beta[i].size(); j++){
                diff_old_new += (old_beta[i][j] - beta[i_l0][i_l1][i_l2][i][j])*(old_beta[i][j] - beta[i_l0][i_l1][i_l2][i][j]);
                beta_old_norm += old_beta[i][j]*old_beta[i][j];
            }
        }
        diff_old_new = diff_old_new/beta_old_norm;
        old_beta = beta[i_l0][i_l1][i_l2];
        current_iter++;

    }
    return obj;
}


double Solver::BCD_sigmoid(int i_l0, int i_l1, int i_l2){
    int MAXITER = 50;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }

    double diff_old_new = INFINITY;
    int current_iter = 0;

    auto old_beta = std::make_shared<std::vector<std::vector<double> > >(beta[i_l0][i_l1][i_l2]);

    double obj=0;


    auto y_tilde = std::make_shared<std::vector<double> >(n,0);
    
    for(int i =0; i < n; i++){
        double xtbeta = 0;
          
        for(int j = 0; j < r; j++){
            xtbeta += old_beta->at(j)[X->at(i)[j]];
        }
        if(!(X_cont == nullptr)){
             
            for(int j = 0; j < old_beta->at(r).size();j++){
                xtbeta += old_beta->at(r)[j]*X_cont->at(i)[j];
            }
        }
        //remove intercept
        
        xtbeta += old_beta->at(intercept_index)[0];
        
        y_tilde->at(i) = xtbeta + 2.0*y_cat->at(i)/(1.0 + std::exp(y_cat->at(i)*xtbeta));
        obj += std::log(1 + std::exp(-y_cat->at(i)*xtbeta));
    }
    obj /= n;
    obj += penalty(i_l0,  i_l1,  i_l2);
    if(VERBOSE){
            log_file << "Starting BCD with l0,l1,l2= " << lambda0[i_l0] << " , " << lambda1[i_l1] << " , " << lambda2[i_l2] << " with penalty " << penalty(i_l0,  i_l1,  i_l2) << std::flush << std::endl;
    }



    while(current_iter <= MAXITER && diff_old_new > TOL){
        if(VERBOSE){
            log_file << "Iteration " << current_iter << " Objective : " << obj << std::flush << std::endl;
        }

        Solver* solver;
        auto l0_vec = std::vector<double>(1,4*lambda0[i_l0]);
        auto l1_vec = std::vector<double>(1,4*lambda1[i_l1]);
        auto l2_vec = std::vector<double>(1,4*lambda2[i_l2]);
        if(X_cont == nullptr){
            solver = new Solver(X,y_tilde, l0_vec,l1_vec,l2_vec,old_beta);
        }
        else{
            solver = new Solver(X,X_cont,y_tilde,l0_vec,l1_vec,l2_vec,old_beta);
        }
        solver->BCD(0,0,0);
        beta[i_l0][i_l1][i_l2] = solver->beta[0][0][0];

        delete solver;
        ////////////////////////////

        //////See how much progress we made
        diff_old_new = 0;
        double beta_old_norm = 0;
        for(int i = 0; i < old_beta->size()-1; i++){
            for(int j = 0; j < old_beta->at(i).size(); j++){
                diff_old_new += (old_beta->at(i)[j] - beta[i_l0][i_l1][i_l2][i][j])*(old_beta->at(i)[j] - beta[i_l0][i_l1][i_l2][i][j]);
                beta_old_norm += old_beta->at(i)[j]*old_beta->at(i)[j];
            }
        }
        diff_old_new = diff_old_new/beta_old_norm;
        old_beta = std::make_shared<std::vector<std::vector<double> > >(beta[i_l0][i_l1][i_l2]);
        current_iter++;

        obj = 0;
        for(int i =0; i < n; i++){
            double xtbeta = 0;
            for(int j = 0; j < r; j++){
                xtbeta += old_beta->at(j)[X->at(i)[j]];
            }
            if(!(X_cont == nullptr)){
                for(int j = 0; j < old_beta->at(r).size();j++){
                    xtbeta += old_beta->at(r)[j]*X_cont->at(i)[j];
                }
            }
            //remove intercept
            xtbeta += old_beta->at(intercept_index)[0];
            y_tilde->at(i) = xtbeta + 2.0*y_cat->at(i)/(1.0 + std::exp(y_cat->at(i)*xtbeta));
            obj += std::log(1 + std::exp(-y_cat->at(i)*xtbeta));
        }
        obj /= n;
        obj += penalty(i_l0,  i_l1,  i_l2);
        
    }
    return obj;
}

double Solver::penalty(int i_l0, int i_l1, int i_l2){
    double res = 0;
    for(int i = 0; i < r; i++){
        std::unordered_set<double> unique_vals;
        for(double v : beta[i_l0][i_l1][i_l2][i]){
            
            unique_vals.insert(v);
            if(abs(v) > TOL){
                res += lambda0[i_l0];
            }
            res += lambda2[i_l2]*v*v;
            
        }
        res += lambda1[i_l1]*unique_vals.size();
    }
    
    return res;
}
double Solver::MSE(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<double> & yval){
    double mse = 0;
    double norm_yval = 0;
    for(int i = 0; i < Xval.size(); i++){
        double curr_error =  yval[i];
        for(int j = 0; j < r; j++){
            curr_error -= beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        curr_error -= beta[i_l0][i_l1][i_l2][r][0];
        mse += curr_error*curr_error;
        norm_yval += yval[i]*yval[i];
    }
    return mse/norm_yval;
}
double Solver::MSE(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<double> & yval){
    double mse = 0;
    double norm_yval = 0;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }
    for(int i = 0; i < Xval.size(); i++){
        double curr_error =  yval[i];
        for(int j = 0; j < r; j++){
            curr_error -= beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        if (!(X_cont == nullptr)){
            for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size(); j++){
                curr_error -= beta[i_l0][i_l1][i_l2][r][j]* Xcontval[i][j];
            }
        }
        curr_error -= beta[i_l0][i_l1][i_l2][intercept_index][0];
        mse += curr_error*curr_error;
        norm_yval += yval[i]*yval[i];
    }
    return mse/norm_yval;
}
std::vector<std::vector<double> > Solver::find_best_beta(std::vector<std::vector<int> > & Xval, std::vector<double> & yval){
    int best_l0,best_l1,best_l2;
    double best_error = INFINITY;
    for(int i_l2 = 0; i_l2 < lambda2.size(); i_l2++){ //Ridge parameter
        for(int i_l1 = 0; i_l1 < lambda1.size(); i_l1++){ //Clustering parameter
            for(int i_l0 = 0; i_l0 < lambda0.size(); i_l0++){ //L0 parameter
                    double obj = BCD(i_l0,i_l1,i_l2);
                    double mse = MSE(i_l0,i_l1,i_l2,Xval,yval);
                    if(VERBOSE){
                        log_file << "Objective : " << obj << " Validation mse " << mse << std::flush << std::endl;
                    }
                    if (mse < best_error){
                        best_l0 = i_l0;
                        best_l1 = i_l1;
                        best_l2 = i_l2;
                        best_error = mse;
                    }
                    if (WARM_START && lambda0.size() > 1 && i_l0 < lambda0.size()-1){
                        beta[i_l0+1][i_l1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
                    else if(WARM_START && lambda0.size() == 1 && i_l1 < lambda1.size()-1){
                        beta[i_l0][i_l1+1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
            }
        }
    }
  
    return beta[best_l0][best_l1][best_l2];
}
std::vector<std::vector<double> > Solver::find_best_beta(std::vector<std::vector<int> > & Xval, std::vector<int> & yval,std::string metric){
    int best_l0,best_l1,best_l2;
    double best_error = INFINITY;
    for(int i_l2 = 0; i_l2 < lambda2.size(); i_l2++){ //Ridge parameter
        for(int i_l1 = 0; i_l1 < lambda1.size(); i_l1++){ //Clustering parameter
            for(int i_l0 = 0; i_l0 < lambda0.size(); i_l0++){ //L0 parameter
                    double obj = BCD_sigmoid_prox(i_l0,i_l1,i_l2);
                    double error_val;
                    if (metric == "accuracy"){
                        error_val = 1.0 - Accuracy(i_l0,i_l1,i_l2,Xval,yval);
                    }
                    else{
                        error_val = log_loss(i_l0,i_l1,i_l2,Xval,yval);
                    }
                    if(VERBOSE){
                        log_file << "Objective : " << obj << " Validation error " << error_val << std::flush << std::endl;
                    }
                    if (error_val < best_error){
                        best_l0 = i_l0;
                        best_l1 = i_l1;
                        best_l2 = i_l2;
                        best_error = error_val;
                    }
                    if (WARM_START && lambda0.size() > 1 && i_l0 < lambda0.size()-1){
                        beta[i_l0+1][i_l1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
                    else if(WARM_START && lambda0.size() == 1 && i_l1 < lambda1.size()-1){
                        beta[i_l0][i_l1+1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
            }
        }
    }
  
    return beta[best_l0][best_l1][best_l2];
}
std::vector<std::vector<double> > Solver::find_best_beta(std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval , std::vector<double> & yval){
    int best_l0,best_l1,best_l2;
    double best_error = INFINITY;
    for(int i_l2 = 0; i_l2 < lambda2.size(); i_l2++){ //Ridge parameter
        for(int i_l1 = 0; i_l1 < lambda1.size(); i_l1++){ //Clustering parameter
            for(int i_l0 = 0; i_l0 < lambda0.size(); i_l0++){ //L0 parameter
                    double obj = BCD(i_l0,i_l1,i_l2);
                    double mse = MSE(i_l0,i_l1,i_l2,Xval,Xcontval,yval);
                    if(VERBOSE){
                        log_file << "Objective : " << obj << " Validation mse " << mse << std::flush << std::endl;
                    }
                    if (mse < best_error){
                        best_l0 = i_l0;
                        best_l1 = i_l1;
                        best_l2 = i_l2;
                        best_error = mse;
                    }
                    if (WARM_START && lambda0.size() > 1 && i_l0 < lambda0.size()-1){
                        beta[i_l0+1][i_l1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
                    else if(WARM_START && lambda0.size() == 1 && i_l1 < lambda1.size()-1){
                        beta[i_l0][i_l1+1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
            }
        }
    }
    return beta[best_l0][best_l1][best_l2];
}


std::vector<std::vector<double> > Solver::find_best_beta(std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval , std::vector<int> & yval,std::string metric){
    int best_l0,best_l1,best_l2;
    double best_error = INFINITY;
    for(int i_l2 = 0; i_l2 < lambda2.size(); i_l2++){ //Ridge parameter
        for(int i_l1 = 0; i_l1 < lambda1.size(); i_l1++){ //Clustering parameter
            for(int i_l0 = 0; i_l0 < lambda0.size(); i_l0++){ //L0 parameter
                    double obj = BCD_sigmoid_prox(i_l0,i_l1,i_l2);
                    double error_val;
                    if(metric == "accuracy"){
                        error_val = 1.0 - Accuracy(i_l0,i_l1,i_l2,Xval,Xcontval,yval);
                    }
                    else{
                        error_val = log_loss(i_l0,i_l1,i_l2,Xval,Xcontval,yval);
                    }

                    
                    if(VERBOSE){
                        log_file << "Objective : " << obj << " Validation error " << error_val << std::flush << std::endl;
                    }
                    if (error_val < best_error){
                        best_l0 = i_l0;
                        best_l1 = i_l1;
                        best_l2 = i_l2;
                        best_error = error_val;
                    }
                    if (WARM_START && lambda0.size() > 1 && i_l0 < lambda0.size()-1){
                        beta[i_l0+1][i_l1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
                    else if(WARM_START && lambda0.size() == 1 && i_l1 < lambda1.size()-1){
                        beta[i_l0][i_l1+1][i_l2] = beta[i_l0][i_l1][i_l2];
                    }
            }
        }
    }
    return beta[best_l0][best_l1][best_l2];
}

double Solver::Accuracy(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<int> & yval){
    int correct_predictions = 0;
    for(int i = 0; i < Xval.size(); i++){
        double xtb =  0.0;
        for(int j = 0; j < r; j++){
            xtb += beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        xtb += beta[i_l0][i_l1][i_l2][r][0];
        
        if(yval[i]*xtb > 0){
            correct_predictions++;
        }
    }
    return ((double)correct_predictions) / Xval.size();
}

double Solver::log_loss(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<int> & yval){
    double log_loss = 0.0;
    for(int i = 0; i < Xval.size(); i++){
        double xtb =  0.0;
        for(int j = 0; j < r; j++){
            xtb += beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        xtb += beta[i_l0][i_l1][i_l2][r][0];

        double probability = 1.0/(1.0 + std::exp(-xtb));
        
        if(yval[i] > 0){
            log_loss -= std::log(probability);
        }
        else{
            log_loss -= std::log(1-probability);
        }
    }
    return (log_loss) / Xval.size();
}

double Solver::log_loss(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<int> & yval){
    double log_loss = 0;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }

    for(int i = 0; i < Xval.size(); i++){
        double xtb =  0.0;
        for(int j = 0; j < r; j++){
            xtb += beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        if (!(X_cont == nullptr)){
            for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size(); j++){
                xtb += beta[i_l0][i_l1][i_l2][r][j]* Xcontval[i][j];
            }
        }
        xtb += beta[i_l0][i_l1][i_l2][intercept_index][0];
        
        double probability = 1.0/(1 + std::exp(-xtb));
        
        if(yval[i] > 0){
            log_loss -= std::log(probability);
        }
        else{
            log_loss -= std::log(1-probability);
        }
    }
    return (log_loss) / Xval.size();
}

double Solver::Accuracy(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<int> & yval){
    int correct_predictions = 0;
    int intercept_index;
    if (X_cont == nullptr){
        intercept_index = r;
    }
    else {
        intercept_index = r+1;
    }

    for(int i = 0; i < Xval.size(); i++){
        double xtb =  0.0;
        for(int j = 0; j < r; j++){
            xtb += beta[i_l0][i_l1][i_l2][j][Xval[i][j]];
        }
        if (!(X_cont == nullptr)){
            for(int j = 0; j < beta[i_l0][i_l1][i_l2][r].size(); j++){
                xtb += beta[i_l0][i_l1][i_l2][r][j]* Xcontval[i][j];
            }
        }
        xtb += beta[i_l0][i_l1][i_l2][intercept_index][0];
        
        if(yval[i]*xtb > 0){
            correct_predictions++;
        }
    }
    return ((double)correct_predictions) / Xval.size();
}