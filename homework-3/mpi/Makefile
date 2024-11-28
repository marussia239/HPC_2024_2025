# mpirun -np 10 ./hw3

all: hw3

hw3: pi_homework.c
	mpiicx -O3 -xHost -qopt-report3 pi_homework.c -o hw3

clean:
	rm hw3