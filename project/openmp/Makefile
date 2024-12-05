

all: mandelbrot

mandelbrot: mandelbrot.cpp
	icpx -qopenmp -xHost -O3 -qopt-report3 mandelbrot.cpp -o mandelbrot

run:
	./mandelbrot out.csv

clean: 
	rm mandelbrot