

all: mandelbrot

mandelbrot: mandelbrot.cpp
	icpx -qopenmp -xHost -O3 -qopt-report3 mandelbrot.cpp -o mandelbrot

run:
	./mandelbrot out.csv

clean: 
	rm mandelbrot &
	rm mandelbrot.opt.ld.yaml &
	rm mandelbrot.opt.yaml &
	rm mandelbrot.optrpt &
	rm ipo_out.optrpt &