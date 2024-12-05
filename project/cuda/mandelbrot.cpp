#include <iostream>
#include <fstream>
#include <cuda/std/complex>
#include <chrono>

#include <cuda.h>
#include "cuda_runtime.h"

// Ranges of the set
#define MIN_X -2
#define MAX_X 1
#define MIN_Y -1
#define MAX_Y 1

// Image ratio
#define RATIO_X (MAX_X - MIN_X)
#define RATIO_Y (MAX_Y - MIN_Y)

// Image size
#define RESOLUTION 10000
#define WIDTH (RATIO_X * RESOLUTION)
#define HEIGHT (RATIO_Y * RESOLUTION)

#define STEP ((double)RATIO_X / WIDTH)

#define DEGREE 2        // Degree of the polynomial
#define ITERATIONS 1500 // Maximum number of iterations

using namespace std;

__global__ void calculateMandelbrot(int *image);

int main(int argc, char **argv)
{
    // int *const image = new int[HEIGHT * WIDTH];
    int *image;
    cudaMalloc((void **)&image, HEIGHT * WIDTH * sizeof(int));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(
        (WIDTH  + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    const auto start = chrono::steady_clock::now();

    calculateMandelbrot<<<numBlocks, threadsPerBlock>>>(image);
    cudaDeviceSynchronize();
    
    const auto end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " seconds." << endl;

    // Write the result to a file
    ofstream matrix_out;

    if (argc < 2)
    {
        cout << "Please specify the output file as a parameter." << endl;
        cudaFree(image);
        return -1;
    }

    matrix_out.open(argv[1], ios::trunc);
    if (!matrix_out.is_open())
    {
        cout << "Unable to open file." << endl;
        cudaFree(image);
        return -2;
    }

    int *imageCpu = new int[HEIGHT * WIDTH];
    cudaMemcpy(imageCpu, image, WIDTH * HEIGHT * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(image);

    const auto print_start = chrono::steady_clock::now();
    for (int row = 0; row < HEIGHT; row++)
    {
        for (int col = 0; col < WIDTH; col++)
        {
            matrix_out << imageCpu[row * WIDTH + col];

            if (col < WIDTH - 1)
                matrix_out << ',';
        }
        if (row < HEIGHT - 1)
            matrix_out << endl;
    }
    matrix_out.close();
    const auto print_end = chrono::steady_clock::now();
    cout << "Time elapsed: "
         << chrono::duration_cast<chrono::milliseconds>(print_end - print_start).count()
         << " seconds." << endl;

    delete imageCpu;
    return 0;
}

__global__ void calculateMandelbrot(int *image) {
    int col_intervals = WIDTH  / (gridDim.x * blockDim.x);
    int row_intervals = HEIGHT / (gridDim.y * blockDim.y);

    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    
    int col_start = col_intervals * threadX;
    int row_start = row_intervals * threadY;

    int col_end = col_intervals * (threadX + 1) - 1;
    int row_end = row_intervals * (threadY + 1) - 1;

    for (int row = row_start; row < row_end; row++) {
        for (int col = col_start; col < col_end; col++) {
            int pos = row * WIDTH + col;
            image[pos] = 0;

            const cuda::std::complex<double> c(col * STEP + MIN_X, row * STEP + MIN_Y);

            // z = z^2 + c
            cuda::std::complex<double> z(0, 0);
            for (int i = 1; i <= ITERATIONS; i++)
            {
                z = pow(z, 2) + c;

                // If it is convergent
                if (abs(z) >= 2)
                {
                    image[pos] = i;
                    break;
                }
            }
        }
    }
}