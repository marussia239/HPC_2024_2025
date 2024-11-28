# Parameters of the tests

Number of intervals: 100000000000
Execution time measured with clock() from time.h.

# Sequential case

Compile command: `icx -O3 -xHost -qopt-report3 pi_homework.c -o hw3`
Execution time (s):
- 58.20 in total.

# Parallel case with MPI

Compile command: `icx -O3 -xHost -qopenmp -qopt-report3 omp_homework_par.c -o h1_par`
Execute command: ``


Execution time (s) with 11 processes (10 slaves, 1 master):
- 23.60 in total.
Execution time (s) with 21 processes (10 slaves, 1 master):
- 9.78 in total.
Execution time (s) with 41 processes (10 slaves, 1 master):
- 5.49 in total.
Execution time (s) with 81 processes (10 slaves, 1 master):
- 2.95 in total.


