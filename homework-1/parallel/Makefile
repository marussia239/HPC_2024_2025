# source /opt/intel/oneapi/setvars.sh

all: h1_par

h1_par: omp_homework_par.c
	icc -O3 -xHost -qopenmp -qopt-report5 omp_homework_par.c -o h1_par

clean:
	rm h1_par
	rm omp_homework_par.optrpt