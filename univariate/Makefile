
CC = g++
SOFLAGS = -shared

EIGENPATH = /home/software/eigen3/3.3.7/include/eigen3/
#EIGENPATH = /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/ #Path for MAC Os

DEBUGFLAGS = -g -Wall  -fsanitize=address -libasan

CFLAGS = -fPIC -std=c++17 -I $(EIGENPATH)

 
# ****************************************************
 
main: main.o Quadratic.o SegSolverCore.o PWQclass.o Solver.o
	$(CC) $(CFLAGS)  -o main.out main.o Quadratic.o SegSolverCore.o PWQclass.o Solver.o

lib: interface.o Quadratic.o SegSolverCore.o PWQclass.o Solver.o
	$(CC) $(SOFLAGS)  -o BCD_solver.so interface.o Quadratic.o SegSolverCore.o PWQclass.o Solver.o 
 
clean:
	rm -f *.o
# ****************************************************
 
main.o: main.cpp
	$(CC) $(CFLAGS) -c main.cpp

Quadratic.o: Quadratic.cpp Quadratic.h
	$(CC) $(CFLAGS) -c Quadratic.cpp

SegSolverCore.o: SegSolverCore.cpp SegSolverCore.h
	$(CC) $(CFLAGS) -c SegSolverCore.cpp 
	$(CC) $(CFLAGS) -c main.cpp

PWQclass.o: PWQclass.cpp PWQclass.h
	$(CC) $(CFLAGS) -c PWQclass.cpp 

interface.o: interface.cpp
	$(CC) $(CFLAGS) -c interface.cpp

Solver.o: Solver.cpp Solver.h
	$(CC) $(CFLAGS)  -c Solver.cpp