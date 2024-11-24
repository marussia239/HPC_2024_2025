#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

// CUDA kernel to compute heat conduction
__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out) {
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < ni - 1 && j > 0 && j < nj - 1) {
        i00 = I2D(ni, i, j);
        im10 = I2D(ni, i - 1, j);
        ip10 = I2D(ni, i + 1, j);
        i0m1 = I2D(ni, i, j - 1);
        i0p1 = I2D(ni, i, j + 1);

        d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
        d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

        temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
    }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;


    // loop over all points in domain (except boundary)
    for ( int j=1; j < nj-1; j++ ) {
        for ( int i=1; i < ni-1; i++ ) {
        // find indices into linear memory
        // for central point and neighbours
        i00 = I2D(ni, i, j);
        im10 = I2D(ni, i-1, j);
        ip10 = I2D(ni, i+1, j);
        i0m1 = I2D(ni, i, j-1);
        i0p1 = I2D(ni, i, j+1);

        // evaluate derivatives
        d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
        d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

        // update temperatures
        temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
        }
    }
}

int main() {
    int istep, nstep, ni, nj, threads_per_block;
    float tfac = 8.418e-5; // thermal diffusivity of silver

    printf("Enter grid dimensions (ni, nj): ");
    scanf("%d %d", &ni, &nj);
    printf("Enter number of timesteps: ");
    scanf("%d", &nstep);
    printf("Enter number of threads per block (e.g., 16): ");
    scanf("%d", &threads_per_block);

    float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;
    float *d_temp1, *d_temp2;

    const int size = ni * nj * sizeof(float);

    temp1_ref = (float*)malloc(size);
    temp2_ref = (float*)malloc(size);
    temp1 = (float*)malloc(size);
    temp2 = (float*)malloc(size);

    // Initialize with random data
    for (int i = 0; i < ni*nj; ++i) {
        temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand() / (float)(RAND_MAX / 100.0f);
    }

    cudaError_t err = cudaMalloc((void**)&d_temp1, size);
    
    if (err != cudaSuccess) {
        printf("CUDA malloc failed! Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMalloc((void**)&d_temp2, size);

    if (err != cudaSuccess) {
        printf("CUDA malloc failed! Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(d_temp1, temp1, size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        printf("CUDA memcpy failed! Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpy(d_temp2, temp2, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess) {
        printf("CUDA memcpy failed! Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    dim3 threadsPerBlock(threads_per_block, threads_per_block);
    dim3 numBlocks((ni + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (nj + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Execute the CPU-only reference version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (istep=0; istep < nstep; istep++) {
        step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

        // swap the temperature pointers
        temp_tmp = temp1_ref;
        temp1_ref = temp2_ref;
        temp2_ref= temp_tmp;   
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
    printf("CPU Execution Time: %.6f seconds\n", cpu_time);

    // Execute the modified version using same data
    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (istep = 0; istep < nstep; istep++) {
        step_kernel_mod<<<numBlocks, threadsPerBlock>>>(ni, nj, tfac, d_temp1, d_temp2);
        cudaDeviceSynchronize();

        temp_tmp = d_temp1;
        d_temp1 = d_temp2;
        d_temp2 = temp_tmp;
    }
    auto end_gpu = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(end_gpu - start_gpu).count();
    printf("GPU Execution Time: %.6f seconds\n", gpu_time);

    cudaMemcpy(temp1, d_temp1, size, cudaMemcpyDeviceToHost);

      float maxError = 0;
    // Output should always be stored in the temp1 and temp1_ref at this point
    for( int i = 0; i < ni*nj; ++i ) {
        if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
    }

    // Check and see if our maxError is greater than an error bound
    if (maxError > 0.0005f)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

    printf("Grid: %dx%d, Timesteps: %d, Threads: %d\n", ni, nj, nstep, threads_per_block);
    printf("CPU Time: %.6f seconds\n", cpu_time);
    printf("GPU Time: %.6f seconds\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("Max Error: %.6f\n", maxError);

    FILE *results_file = fopen("results.csv", "a");
    fprintf(results_file, "%d,%d,%d,%d,%.6f,%.6f,%.2f\n", ni, nj, nstep, threads_per_block, cpu_time, gpu_time, cpu_time / gpu_time);
    fclose(results_file);

    free( temp1_ref );
    free( temp2_ref );
    free( temp1 );
    free( temp2 );
    cudaFree( d_temp1 );
    cudaFree( d_temp2 );

    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    return 0;
}
