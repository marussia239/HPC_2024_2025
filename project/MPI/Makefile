# mpirun -np 10 ./mandelbrot

all: mandelbrot

mandelbrot: mandelbrot.cpp
	mpiicpx -O3 -xHost -qopt-report3 mandelbrot.cpp -o mandelbrot

run:
	mpirun -np 11 ./mandelbrot

clean:
	rm mandelbrot
