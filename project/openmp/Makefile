

all: mandelbrot

mandelbrot: mandelbrot.cpp
	icpx -qopenmp -xHost -O3 -qopt-report3 mandelbrot.cpp -o mandelbrot

clean: 
	rm mandelbrot