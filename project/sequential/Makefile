

all: mandelbrot

mandelbrot: mandelbrot.cpp
	icpx -xHost -O3 -qopt-report3 mandelbrot.cpp -o mandelbrot

run:
	./mandelbrot out.csv

clean: 
	rm mandelbrot