
#include "SegSolverCore.h"
#include <vector>
#include <set>
#include <iostream>
#include <math.h>
#include <memory>


SegSolver::SegSolver(std::vector<double> y_, double l0_, double l1_,double l2_){
  y = y_;
  l0 = l0_;
  l2 = l2_;
  if (y.size() > 1){
    l1 = l1_;
    beta = std::vector<double>(y.size(),0.0);
    for (int i=0; i < y.size();i++){
        std::set<Quadratic> knots;

        knots.insert(Quadratic(0,l0,(-1.0/2)-l2,1.0*y[i],-y[i]*y[i]/2.0));
        knots.insert(Quadratic(INFINITY,0,(-1.0/2)-l2,1.0*y[i],-y[i]*y[i]/2.0));
  
        auto ff = std::make_shared<PWQclass>(2,knots);
        

        e.push_back(ff); //Make e_i
        if (i == 0){
            delta.push_back(ff);
            auto temp = std::make_shared<PWQclass>();
            f.push_back(temp);
        }
        else{//Make delta, f
                auto new_res = delta[i-1]->maximize_flood(l1);
                //f_new->print_knots();
                double b_new = new_res.second.first;
                double y_star_new = new_res.second.second;
                f.push_back( new_res.first);
                auto delta_f = std::make_shared<PWQclass>( new_res.first, ff);
                delta.push_back(delta_f);
                b_.push_back(b_new);
                y_star.push_back(y_star_new);
                
        }
        
    }
  }
}
SegSolver::SegSolver(std::vector<double> y_,std::vector<double> weights, double l0_, double l1_,double l2_){
  y = y_;
  l0 = l0_;
  l2 = l2_;
  w0 = weights[0];
  if (y.size() > 1){
    l1 = l1_;
    beta = std::vector<double>(y.size(),0.0);
    for (int i=0; i < y.size();i++){
        
        std::set<Quadratic> knots;
        

        knots.insert(Quadratic(0,l0,(-weights[i]/2) - l2,weights[i]*y[i],-weights[i]*y[i]*y[i]/2.0));
        knots.insert(Quadratic(INFINITY,0,(-weights[i]/2) - l2,weights[i]*y[i],-weights[i]*y[i]*y[i]/2.0));
  
        auto ff = std::make_shared<PWQclass>(2,knots);
        

        e.push_back(ff); //Make e_i

        if (i == 0){
            delta.push_back(ff);
            auto temp = std::make_shared<PWQclass>();
            f.push_back(temp);
        }
        else{//Make delta, f
              
                auto new_res = delta[i-1]->maximize_flood(l1);
                //f_new->print_knots();
                double b_new = new_res.second.first;
                double y_star_new = new_res.second.second;
                f.push_back(new_res.first);
                auto delta_f = std::make_shared<PWQclass>( new_res.first,  ff);
                
                delta.push_back(delta_f);
                b_.push_back(b_new);
                y_star.push_back(y_star_new);
        }
        
    }
  }
}
std::vector<double> SegSolver::backtrace(){ //Do the backtracing procedure
    if (y.size()==1){
          if (.5*abs_c(w0*y[0])*abs_c(w0*y[0])>l0) {return y;}
          else {return std::vector<double>(1,0);}
    }
    for(int i = y.size()-1; i >= 0; i--){
        if(i == y.size()-1){
            auto res_new = delta[i]->maximize_flood(0);
            beta[i] = res_new.second.first;
            continue;
        }
        double y_d = delta[i]->evaluate(beta[i+1]);
        //std::cout << "eval " << i <<" " <<y_d <<" " <<b_[i] <<" " << y_star[i]  << std::endl;
        if(y_d < y_star[i] - l1){
          beta[i] = b_[i];
        }
        else {beta[i] = beta[i+1];}
    }

    return beta;
}
