#include "SegSolverCore.h"
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <math.h>
#include <utility>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <ctime>


#define WARM_START true
#define TOL 1e-4

class Solver{
    public:
        std::vector<std::vector<std::vector< std::vector<std::vector<double> > > > > beta;
        std::shared_ptr<std::vector<std::vector<double> > > beta0;
        int r; //number of groups
        int n;       
        int p; //dimension of dummified X
        std::vector<double> lambda0,lambda1,lambda2;
        std::shared_ptr<std::vector<std::vector<int> > > X; //X is of dimension n*r
        std::shared_ptr<std::vector<std::vector<double> > > X_cont; 
        Eigen::MatrixXd X_cont_m;

        std::shared_ptr<std::vector<double> >  y;
        std::shared_ptr<std::vector<int> >  y_cat;

        std::ofstream log_file;
        bool VERBOSE;

        Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<double> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_=false);
        Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<std::vector<double> > > & X_cont_ ,std::shared_ptr<std::vector<double> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_=false );
        
        //categorical y
        Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<int> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_=false );
        Solver(std::shared_ptr<std::vector<std::vector<int> > > & X_,std::shared_ptr<std::vector<std::vector<double> > > & X_cont_ ,std::shared_ptr<std::vector<int> > & y_, std::vector<double> & l0,std::vector<double> & l1,std::vector<double> & l2, std::shared_ptr<std::vector<std::vector<double> > > beta0_,bool verbose_=false);

        double BCD(int i_l0, int i_l1, int i_l2);
        double BCD_sigmoid_prox(int i_l0, int i_l1, int i_l2);
        double BCD_sigmoid(int i_l0, int i_l1, int i_l2);

        double MSE(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<double> & yval);
        double MSE(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<double> & yval);

        double Accuracy(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<int> & yval);
        double Accuracy(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<int> & yval);

        double log_loss(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval, std::vector<int> & yval);
        double log_loss(int i_l0, int i_l1, int i_l2, std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval, std::vector<int> & yval);

        std::vector<std::vector<double> > find_best_beta(std::vector<std::vector<int> > & Xval, std::vector<double> & yval);
        std::vector<std::vector<double> > find_best_beta(std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval , std::vector<double> & yval);

        //categorical validation
        std::vector<std::vector<double> > find_best_beta(std::vector<std::vector<int> > & Xval, std::vector<int> & yval,std::string metric ="accuracy");
        std::vector<std::vector<double> > find_best_beta(std::vector<std::vector<int> > & Xval,std::vector<std::vector<double> > & Xcontval , std::vector<int> & yval,std::string metric ="accuracy");

        double penalty(int i_l0, int i_l1, int i_l2);
};