#include "Solver.h"
#include <vector>
#include <iostream>
#include <random>



int main(){
    // random device class instance, source of 'true' randomness for initializing random seed
    std::random_device rd; 

    // Mersenne twister PRNG, initialized with seed from previous random device instance
    std::mt19937 gen(rd()); 
    std::normal_distribution<float> d(0, 1); 
    for(int run = 0; run < 100;run++){

    
    int n = 100;
    int r = 1;
    int rcont = 10;
    std::shared_ptr<std::vector<double> > yptr = std::make_shared<std::vector<double> >();
    std::shared_ptr<std::vector<std::vector<int> > >  X = std::make_shared<std::vector<std::vector<int> > > (n,std::vector<int>(r));
    std::shared_ptr<std::vector<std::vector<double> > >  Xcont = std::make_shared<std::vector<std::vector<double> > > (n,std::vector<double>(rcont));
    for(int i = 0; i < n;i++){
        X->at(i)[0] = i%2;
        for(int rconti = 0; rconti< rcont;rconti++){
            Xcont->at(i)[rconti] = d(gen);
        }
        
        yptr->push_back(d(gen));
    }
    std::vector<double> l0 = {0.0,0.1};
    std::vector<double> l1 = {0.0,0.1};
    std::vector<double> l2 = {0.0,0.1};
    std::shared_ptr<std::vector<std::vector<double> > > beta0 = std::make_shared<std::vector<std::vector<double> > >(r+1);

    beta0->at(0) = std::vector<double>(2,0);
    beta0->at(1) = std::vector<double>(rcont,0);
        
   
    auto solver = Solver(X,Xcont,yptr,l0,l1,l2,beta0);
    auto beta = solver.find_best_beta(*X,*Xcont,*yptr);
   
                    
    for(auto v : beta){
      for(auto x:v){
          std::cout << x << " ";
      }
    }
    }
    return 0;

    /*
    #include <iostream>
#include <cctype>
#include <random>

using u32    = uint_least32_t; 
using engine = std::mt19937;

int main( void )
{
  std::random_device os_seed;
  const u32 seed = os_seed();

  engine generator( seed );
  std::uniform_int_distribution< u32 > distribute( 1, 6 );

  for( int repetition = 0; repetition < 10; ++repetition )
    std::cout << distribute( generator ) << std::endl;
  return 0;
} 
    */
}
