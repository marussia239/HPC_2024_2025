# mpirun -np 10 ./mandelbrot

all: mandelbrot

mandelbrot: mandelbrot.cpp
	mpiicpx -qopenmp -O3 -xHost -qopt-report3 mandelbrot.cpp -o mandelbrot

run: mandelbrot
	mpirun -np 101 ./mandelbrot out.csv

clean:
	rm mandelbrot
