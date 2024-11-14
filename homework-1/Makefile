# source /opt/intel/oneapi/setvars.sh

all: h1

h1: omp_homework.c
	icc -o3 -xHost -qopenmp -qopt-report3 omp_homework.c -o h1

clean:
	rm h1
	rm omp_homework.opt.yaml
	rm omp_homework.optrpt