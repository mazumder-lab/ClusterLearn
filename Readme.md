

## Requirements

The following python packages are required to run ClusterLearn:

```
numpy
pandas
scikit_learn
rpy2
ctypes
gurobipy (only for the exact solver)
``` 
In addition, the following R package is required:

```
CatReg
``` 

## Installation

Either of the following:

```
cd univariate
make lib
make main
``` 
or

```
cd univariate
g++ -fPIC -std=c++17 -c interface.cpp  SegSolverCore.cpp PWQclass.cpp  
g++ -shared -Wl -o proximal_c.so interface.o SegSolverCore.o PWQclass.o 
``` 

## Usage

Please see ```demo.py``` for a tutorial of the approximate solver. Please see ```demo_exact.py``` for a tutorial of the exact solver. 