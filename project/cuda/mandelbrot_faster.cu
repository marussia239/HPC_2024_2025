#include <iostream>
#include <fstream>
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

// #define STEP ((double)RATIO_X / WIDTH)
#define STEP ((double)(MAX_X - MIN_X) / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1000 // Maximum number of iterations


__global__ void mandelbrot(int *image) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int pos = row * WIDTH + col;
    if (pos >= WIDTH * HEIGHT) return;

    image[pos] = 0;

    const double cr = col * STEP + MIN_X;
    const double ci = row * STEP + MIN_Y;

    // z = z^2 + c
    // cuda::std::complex<double> z(0, 0);
    double zr = 0;
    double zi = 0;
    for (int i = 1; i <= ITERATIONS; i++) {
        //z = pow(z, 2) + c;
        double zr2 = zr * zr;
        double zi2 = zi * zi;
        zi = 2 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

        // If it is convergent
        if (zr2 + zi2 >= 4) {
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

    if (cudaMalloc((void **)&image_d, WIDTH * HEIGHT * sizeof(int)) != cudaSuccess) {
    std::cout << "CUDA malloc failed!" << std::endl;
    return -4;
}

    // Set thread configuration
    dim3 numThreads(16, 16);
    dim3 numBlocks(
        (WIDTH + numThreads.x - 1)  / numThreads.x,
        (HEIGHT + numThreads.y - 1) / numThreads.y
    );

    // Call CUDA kernel
    const auto start = std::chrono::steady_clock::now();
    mandelbrot<<<numBlocks, numThreads>>>(image_d);
    cudaDeviceSynchronize();

    cudaError status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::cout << "Error after kernel execution: " << cudaGetErrorString(status) << " (code " << status << ")" << std::endl;
        delete[] image;
        cudaFree(image_d);
        return -3;
    }

    // Transfer results to host
    cudaMemcpy(image, image_d, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);

    if (cudaMemcpy(image, image_d, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cout << "CUDA memcpy failed!" << std::endl;
        return -5;
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
         << " ms." << std::endl;


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

    const auto print_start = std::chrono::steady_clock::now();
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
    const auto print_end = std::chrono::steady_clock::now();
    std::cout << "Time elapsed for printing: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(print_end - print_start).count()
         << " ms." << std::endl;

    delete[] image; // It's here for coding style, but useless
    cudaFree(image_d);
    return 0;
}