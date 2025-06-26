#include "Quadratic.h"

 bool operator<(const Quadratic& quad, const Quadratic& other_quad) 
    {
        return quad.knot < other_quad.knot;
    }