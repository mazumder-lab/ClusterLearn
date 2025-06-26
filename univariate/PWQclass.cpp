#include "PWQclass.h"
#include <utility>
#include <iostream>
#include <limits>
#include <math.h>
#include <set>
#include <memory>

//const double INFINITY= std::numeric_limits<double>::infinity();
 double mINFINITY= -1*std::numeric_limits<double>::infinity();
 double tol = 0.00001;

double abs_c(double x){
    if (x >= 0) return x;
    else return -x;
}

PWQclass::PWQclass(int n_knots_,std::set<Quadratic>  knots_){
                   
    n_knots = n_knots_;
    knots = knots_;
    //fill_max();
}

void PWQclass::fill_max(){
    int i = 0;
    double prev_knot,prev_spike;
    max_y = mINFINITY;
    for(auto it= knots.begin(); it != knots.end(); it++){
        std::pair<double,double> result;
        if (i==0){ //first knot
            result =maximize_q(it->a,it->b,it->c, mINFINITY,it->knot,  0,it->spike);
        }
        else{
            result =maximize_q(it->a,it->b,it->c, prev_knot,it->knot, prev_spike,it->spike);
        }
        prev_knot = it->knot;
        prev_spike = it->spike;
        if (result.second >= max_y){
            max_y = result.second;
            max_x = result.first;
        }
        i++;
    }
}

PWQclass::PWQclass(){
    n_knots = 0;
    knots = std::set<Quadratic> ();
}
PWQclass::PWQclass(std::shared_ptr<PWQclass>  f1, std::shared_ptr<PWQclass>  f2){
    knots = std::set<Quadratic> ();
    add_pwq(std::move(f1),std::move(f2));
    //fill_max();
}

void PWQclass::add_pwq(std::shared_ptr<PWQclass>  f1, std::shared_ptr<PWQclass>  f2){ //f2 is a quadtratic with one piece only
    
    int n1 = f1->n_knots;
    int n2 = f2->n_knots;

    //Add to f1 f2 parts (ignoring any spike)
    n_knots = n1;
    //knots = f1->knots; //Vector is copied
    
    auto new_quad = f2->knots.begin();
    for(auto it= f1->knots.begin(); it != f1->knots.end(); it++){
        knots.insert(Quadratic(it->knot,it->spike, it->a + new_quad->a,it->b + new_quad->b,it->c + new_quad->c ));
    }
    //handle possile spike in f2
    if(n2 == 2){
        auto new_quad = f2->knots.begin();
        auto it = knots.lower_bound(*new_quad);
        if(it->knot == new_quad->knot){ //not already exists, just add spike
            it->spike += new_quad->spike;
        }
        else{
            auto quad_to_insert = Quadratic(*it);
            quad_to_insert.knot = new_quad->knot;
            quad_to_insert.spike = new_quad->spike;
            knots.insert(quad_to_insert);
        }

    }
}


  std::pair<double,double> PWQclass::maximize_q(double a,double b, double c, double m, double M, double y_sp1, double y_sp2){
     // Maximize a quadratic function over the interval [m,M]
    //Spikes can happen in m and M with values y_sp1, y_sp2
    double xmax,ymax;
    ymax = mINFINITY;
    xmax = m;

    if (m == mINFINITY){
       if (a > 0){
           xmax = m;
           ymax = INFINITY;
       } 
    }
    else{
        double val = a*m*m + b*m +c + y_sp1;
        if (val > ymax) {
            ymax = val;
            xmax = m;
        }
    }

    if (M == INFINITY){
        if (a > 0){
           xmax = M;
           ymax = INFINITY;
       }
    }
    else{
        double val = a*M*M + b*M +c + y_sp2;
        if (val > ymax) {
            ymax = val;
            xmax = M;
        }
    }

    if (a != 0){
        double x = -b/(2*a); // The maximizer of the quadratic function
        if (x<M && x>m && a*x*x + b*x + c > ymax){ // Is this solution feasible?
                xmax = x;
                ymax = a*x*x + b*x + c;
        }
    }

    return std::pair<double,double>(xmax,ymax);
 }

  std::shared_ptr<PWQclass> PWQclass::flood(double a,double b, double c, double m, double M, double y_sp, double T){
    //Threshold a quadratic function over (m,M] with a possible spike at M. T is the threshold level.
    // The flooded function is a PWQ.
    std::shared_ptr<PWQclass>  f = std::make_shared<PWQclass>();
    double c0 = c-T;
    if(a==0 && b==0){
        double y_d = c;
        if (y_d < T){ // Below T
            f->insert_quad(M,0,0,0,T);
            }
        else { //Above T
            f->insert_quad(M,0,0,b, c);}
    }
    else if(a==0){
        double x = -c0/b;
        double y_d = b*m + c;
        if (x > M || x < m){
            if (y_d < T){ // Below T
                f->insert_quad(M,0,0,0, T);
                
                 }
            else { //Above T
            f->insert_quad(M,0,0,b, c);
                
            }
        } else{
            if (y_d< T){
                f->insert_quad(x,0,0,0, T);
                f->insert_quad(M,0,0,b, c);
            }
            else{
                f->insert_quad(x,0,0,b, c);
                f->insert_quad(M,0,0,0, T);
            }
        }
    }
    else{
        double delta = b*b - 4*a*c0;
        if (delta <=0){
            if(a >0){
                f->insert_quad(M,0,a,b,c);
            }
            else {f->insert_quad(M,0,0,0,T);}
        }
        else{
            double x1 = (-b-sqrt(delta))/(2*a);
            double x2 = (-b+sqrt(delta))/(2*a);
            if (x1> x2){
                double temp = x1;
                x1 = x2;
                x2 = temp;
            }
            double knots0[2];
            double a_return0[3];
            double b_return0[3];
            double c_return0[3];
            knots0[0] =x1;
            knots0[1] =x2;

            if (a<0) // Is the function convex or concave.
            {   
                a_return0[0] = 0;
                b_return0[0]= 0;
                c_return0[0]= T;

                a_return0[1] = a;
                b_return0[1]= b;
                c_return0[1]= c;
                
                a_return0[2] = 0;
                b_return0[2]= 0;
                c_return0[2]= T;

            }
            else
            {   

                a_return0[0] = a;
                b_return0[0]= b;
                c_return0[0]= c;

                a_return0[1] = 0;
                b_return0[1]= 0;
                c_return0[1]= T;
                
                a_return0[2] = a;
                b_return0[2]= b;
                c_return0[2]= c;

            }
                

            if (knots0[1]<=m){ //No knot falls in the interval

                    f->insert_quad(M,0,a_return0[2],b_return0[2],c_return0[2]);
            }      
            else if (knots0[0]>=M){//No knot falls in the interval
                    f->insert_quad(M,0,a_return0[0],b_return0[0],c_return0[0]);
            }      
            else if (knots0[0]<= m && knots0[1]<M){ // One knot falls in the interval
                    f->insert_quad(knots0[1],0,a_return0[1],b_return0[1],c_return0[1]);
                    f->insert_quad(M,0,a_return0[2],b_return0[2],c_return0[2]);

            }
            else if (knots0[0]<= m && knots0[1]>=M){
                f->insert_quad(M,0,a_return0[1],b_return0[1],c_return0[1]);
            }
            else if (knots0[0]> m && knots0[1]>=M){
                    f->insert_quad(knots0[0],0,a_return0[0],b_return0[0],c_return0[0]);
                    f->insert_quad(M,0,a_return0[1],b_return0[1],c_return0[1]);
            }    
            else if (knots0[0]> m && knots0[1]<M){
                    f->insert_quad(knots0[0],0,a_return0[0],b_return0[0],c_return0[0]);
                    f->insert_quad(knots0[1],0,a_return0[1],b_return0[1],c_return0[1]);
                    f->insert_quad(M,0,a_return0[2],b_return0[2],c_return0[2]);


            }
    
        }
    }
    f->n_knots = f->knots.size();
    if (y_sp > 0 && a*M*M + b*M+ c + y_sp > T ) {//Does the spike go over T? 
        //f->sp_x.insert(f->sp_x.end(),M);
        auto last_quad = f->knots.rbegin();
        last_quad->spike += a*M*M+b*M+c + y_sp- (last_quad->a*M*M+last_quad->b*M+last_quad->c); //Remove the offset of the quadratic value.
    }
    
    
    return  std::move(f);
}

/*
def evaluate(self, x): #Evaluate the function at x
        y = 0
        
        for i in range(self.sp_n):
            if np.abs_c(self.sp_x[i] - x)<1e-6:
                y = y + self.sp_y[i]
                break
        
        
        if self.n_knots == 0:
            y = y + self.a[0]*x*x + self.b[0]*x + self.c[0]
        elif x<= self.knots[0]:
            y = y + self.a[0]*x*x + self.b[0]*x + self.c[0]
        elif x > self.knots[-1]:
            y = y + self.a[-1]*x*x + self.b[-1]*x + self.c[-1]
        else:
            for i in range(self.n_knots-1):
                if x> self.knots[i] and x <= self.knots[i+1]:
                    y = y + self.a[i+1]*x*x + self.b[i+1]*x + self.c[i+1]
                    break
        return y
*/
double PWQclass::evaluate(double x){
    double y =0;
    auto tofind = Quadratic(x,0,0,0,0);
    auto lb = knots.lower_bound(tofind);
    if(lb->knot == x){
        y += lb->spike;
    }
    y = y + (lb->a)*x*x + (lb->b)*x + (lb->c);           
    return y;

}

std::pair<std::shared_ptr<PWQclass>, std::pair<double,double> > PWQclass::maximize_flood(double lambda1){
    
        fill_max(); //compute the max
    

        std::set<Quadratic> knots_flooded;
        

        
        //Go through intervals defined by critical_merged and put the PWQ function together.
        int j = 0;
        double prev_knot = mINFINITY;
        for(auto it= knots.begin(); it != knots.end(); it++){
            
            std::shared_ptr<PWQclass> f = PWQclass::flood(it->a,it->b,it->c, prev_knot, it->knot,it->spike ,max_y-lambda1);
            
            prev_knot = it->knot;

            knots_flooded.insert(f->knots.begin(),f->knots.end());
            j++;
        }
        int true_len = knots_flooded.size();
        
        auto it = knots_flooded.begin();
        auto nit = it;
        nit++;
        while (nit != knots_flooded.end()){ //Remove redundunt knots. If the function does not actually change over the knot, it is redandunt.
            if (abs_c(it->a - nit->a)<tol && abs_c(it->b - nit->b)<tol && abs_c(it->c - nit->c)<tol && abs_c(it->spike) < tol ){
            
                knots_flooded.erase(it);
                true_len = true_len - 1; 
                it = nit;
                nit++;
            }
            else{
                it++;
                nit++;
            }
        }

        std::shared_ptr<PWQclass> result = std::make_shared<PWQclass>(knots_flooded.size(),knots_flooded);
        //result->fill_max();
        return std::pair<std::shared_ptr<PWQclass>, std::pair<double,double> >( std::move(result), std::pair<double,double>(max_x,max_y));
}
