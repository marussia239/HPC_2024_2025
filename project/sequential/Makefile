

all: mandelbrot

mandelbrot: mandelbrot.cpp
	icpx -xHost -O3 -qopt-report3 mandelbrot.cpp -o mandelbrot

run: mandelbrot
	./mandelbrot out.csv

clean: 
	rm mandelbrot &
	rm mandelbrot.opt.yaml &
	rm mandelbrot.optrpt &