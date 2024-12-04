# Sequential case

The outer loop cannot be vectorized because it's the outer loop. The inner loop cannot be vectorized because every iteration depends on the results of the previous one.

## Hotspot identification

RESOLUTION 5000 and ITERATIONS 1500.

First part of the program (inside loop at line 33) calculates the mandelbrot set values. Sequential time is 126.392 seconds.
Second part of the program (insdie loop at line 77) prints the results to a file. Sequential time is 4.616 seconds.

The first loop is the hotspot.

# Multithread parallelization using OpenMP

Adding `#pragma omp parallel for` before the hotspot loop and running with 20 threads, time is 13.766 second (speedup of about x9.18).