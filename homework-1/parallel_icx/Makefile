# source /opt/intel/oneapi/setvars.sh

all: h1_par

h1_par: omp_homework_par.c
	icx -O3 -xHost -qopenmp -qopt-report3 omp_homework_par.c -o h1_par

clean:
	rm h1_par
	rm omp_homework_par.optrpt
	rm omp_homework_par.opt.yaml