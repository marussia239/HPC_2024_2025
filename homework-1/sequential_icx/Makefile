# source /opt/intel/oneapi/setvars.sh

all: h1_seq

h1_seq: omp_homework_seq.c
	icx -O3 -xHost -qopenmp -qopt-report3 omp_homework_seq.c -o h1_seq

clean:
	rm h1_seq
	rm omp_homework_seq.opt.yaml
	rm omp_homework_seq.optrpt