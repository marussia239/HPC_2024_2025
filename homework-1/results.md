# Parameters of the tests

Size of input array: 100000

# Sequential case

Compile command: `icx -o3 -xHost -qopenmp -qopt-report3 omp_homework_seq.c -o h1_seq`
Execution time (s):
- 37.830 in total, of which:
    - 18.919 for the DFT operation (approx. 50% of total time)
    - 18.911 for the IDFT operation (approx. 50% of total time)

Both DFT and IDFT operations are executed by the same function DFT: DFT is the hotspot.

# Parallel case

Compile command: `icx -O3 -xHost -qopenmp -qopt-report3 omp_homework_par.c -o h1_par`
Execution time (s) with 20 threads (automatically determined: the machine has 20 cores with 1 thread per core):
- 4.840415 in total, of which:
    - 2.437 for the DFT operation (approx. 50% of total time)
    - 2.403 for the IDFT operation (approx. 50% of total time)

