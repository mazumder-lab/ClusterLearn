#include <utility>
#include <iostream>
#include <set>
#include "Quadratic.h"
#include <memory>

class PWQclass{
    public:
        int n_knots;
        double max_x,max_y;
        std::set<Quadratic>  knots;

        PWQclass(int n_knots,std::set<Quadratic >  knots);
        PWQclass(std::shared_ptr<PWQclass> f1, std::shared_ptr<PWQclass> f2);

        PWQclass();
        void fill_max();

        void add_pwq(std::shared_ptr<PWQclass>  f1, std::shared_ptr<PWQclass>  f2);


        std::pair<std::shared_ptr<PWQclass>, std::pair<double,double> > maximize_flood(double lambda1);

        static std::shared_ptr<PWQclass> flood(double a,double b, double c, double m, double M, double y_sp, double T);
        
        double evaluate(double x);

        static std::pair<double,double> maximize_q(double a,double b, double c, double m, double M, double y_sp1, double y_sp2);

        void insert_quad(double knot,double spike, double a, double b,double c){
            Quadratic q(knot,spike,a,b,c);
            knots.insert(q);
        }
        /*
        std::vector<double> get_quad(int i){
            std::vector<double> res(3);
            res[0] = a[i];
            res[1] = b[i];
            res[2] = c[i];
            return res;
        }
        
        void print_knots(){
            std::cout << "knots :";
            for(int i =0; i <n_knots;i++){
                std::cout << knots[i] << "("<< a[i]<< " xx + "<< b[i]<< " x + "<< c[i]<< ")";
            }
            int i = n_knots;
            std::cout << "("<< a[i]<< " xx + "<< b[i]<< " x + "<< c[i]<< ")";
            std::cout << std::endl;
        }
        void print_spikes(){
            std::cout << "spikes : ";
            for(int i = 0; i < sp_n;i++){
                std::cout << sp_x[i] << ","<< sp_y[i] << " ";
            }
            std::cout << std::endl;
        }*/
};

double abs_c(double x);
