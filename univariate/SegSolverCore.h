
#include "PWQclass.h"
#include <vector>
#include <set>
#include <iostream>
#include <math.h>
#include <memory>


class SegSolver {       
  public:             // Access specifier
    double l0,l1,l2;        
    std::vector<double> y,b_,beta,y_star;
    std::vector<std::shared_ptr<PWQclass> > e,f,delta;
    double w0=1;

    SegSolver(std::vector<double> y_, double l0_, double l1_,double l2_);
    
    SegSolver(std::vector<double> y_,std::vector<double> weights, double l0_, double l1_,double l2_);
    
    std::vector<double> backtrace();
};