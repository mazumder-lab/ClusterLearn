

class Quadratic {
    public:
        mutable double knot,spike,a,b,c;
        Quadratic(double knot_,double spike_,double a_,double b_, double c_){
            knot = knot_;
            spike = spike_;
            a = a_;
            b = b_;
            c = c_;
        }
    

};
bool operator<(const Quadratic& quad, const Quadratic& other_quad);
