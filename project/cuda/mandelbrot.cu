#include <iostream>
#include <fstream>
#include <cuda/std/complex>
#include <chrono>

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 1000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations


__global__ void mandelbrot(double *image) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = row * WIDTH + col;
    if (pos >= WIDTH * HEIGHT) return;

    image[pos] = 0;

    const int row = pos / WIDTH;
    const int col = pos % WIDTH;
    const cuda::std::complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

    // z = z^2 + c
    cuda::std::complex<double> z(0, 0);
    for (int i = 1; i <= ITERATIONS; i++) {
        z = pow(z, 2) + c;

        // If it is convergent
        if (abs(z) >= 2) {
            image[pos] = i;
            break;
        }
    }
}


int main(int argc, char **argv) {
    // Init host and device memory
    int *const image = new int[HEIGHT * WIDTH];
    int *image_d;
    cudaMalloc((void **)&image_d, WIDTH * HEIGHT * sizeof(int));

    // Set thread configuration
    dim3 numThreads = (16, 16);
    dim3 numBlocks  = (
        std::ceil(WIDTH  / numThreads.x),
        std::ceil(HEIGHT / numThreads.y)
    );

    // Call CUDA kernel
    mandelbrot<<<numBlocks, numThreads>>>(image_d);

    // Transfer results to host
    cudaMemcpy(image, image_d, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    // Write the result to a file
    std::ofstream matrix_out;

    if (argc < 2) {
        std::cout << "Please specify the output file as a parameter." << std::endl;
        return -1;
    }

    matrix_out.open(argv[1], std::ios::trunc);
    if (!matrix_out.is_open()) {
        std::cout << "Unable to open file." << std::endl;
        return -2;
    }

    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            matrix_out << image[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << std::endl;
    }
    matrix_out.close();

    delete[] image; // It's here for coding style, but useless
    cudaFree(image_d);
    return 0;
}