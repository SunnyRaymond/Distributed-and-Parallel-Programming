/*
 * simulate.cu
 *
 * Implementation of a wave equation simulation, parallelized on the GPU using
 * CUDA.
 *
 * You are supposed to edit this file with your implementation, and this file
 * only.
 *
 */

#include <cstdlib>
#include <iostream>

#include "simulate.hh"

#include <cuda_runtime.h>

using namespace std;


/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 * 
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
**/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}


/* CUDA kernel: perform one time step of the 1D wave equation.
 *
 * d_old:  amplitudes at t-1
 * d_cur:  amplitudes at t
 * d_next: amplitudes at t+1 (output)
 * n:      number of points (i_max)
 */
__global__ void wave_step_kernel(const double *d_old,
                                 const double *d_cur,
                                 double *d_next,
                                 long n)
{
    long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const double c = 0.15;

    if (idx == 0 || idx == n - 1) {
        /* Fixed boundary conditions */
        d_next[idx] = 0.0;
    } else {
        double ai = d_cur[idx];
        d_next[idx] = 2.0 * ai
                    - d_old[idx]
                    + c * (d_cur[idx - 1] - 2.0 * ai + d_cur[idx + 1]);
    }
}


/* Function that will simulate the wave equation, parallelized using CUDA.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 * 
 */
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array) {

    // YOUR CODE HERE

    /* Handle trivial cases on host: nothing to simulate */
    if (t_max <= 0 || i_max <= 0) {
        return current_array;
    }

    /* Device buffers */
    double *d_old  = nullptr;
    double *d_cur  = nullptr;
    double *d_next = nullptr;

    size_t bytes = static_cast<size_t>(i_max) * sizeof(double);

    /* Allocate device memory */
    checkCudaCall(cudaMalloc((void**)&d_old,  bytes));
    checkCudaCall(cudaMalloc((void**)&d_cur,  bytes));
    checkCudaCall(cudaMalloc((void**)&d_next, bytes));

    /* Copy initial generations to device */
    checkCudaCall(cudaMemcpy(d_old,  old_array,     bytes, cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(d_cur,  current_array, bytes, cudaMemcpyHostToDevice));
    // d_next will be written by the kernel; no need to initialize

    /* Configure kernel launch */
    long threads_per_block = block_size;
    if (threads_per_block <= 0) {
        threads_per_block = 256; // sensible default
    }
    long num_blocks_long = (i_max + threads_per_block - 1) / threads_per_block;
    int  num_blocks = static_cast<int>(num_blocks_long);

    /* Time-stepping loop: each iteration launches one kernel over space.
       We rotate the device pointers after every step. */
    for (long t = 0; t < t_max; ++t) {
        wave_step_kernel<<<num_blocks, (int)threads_per_block>>>(d_old, d_cur, d_next, i_max);
        checkCudaCall(cudaGetLastError());

        // Rotate roles: old <- cur, cur <- next, next <- old
        double *tmp = d_old;
        d_old  = d_cur;
        d_cur  = d_next;
        d_next = tmp;
    }

    /* Ensure all kernels have finished before copying results back */
    checkCudaCall(cudaDeviceSynchronize());

    /* Copy final result (current generation) back to host buffer */
    checkCudaCall(cudaMemcpy(current_array, d_cur, bytes, cudaMemcpyDeviceToHost));

    /* Clean up device memory */
    checkCudaCall(cudaFree(d_old));
    checkCudaCall(cudaFree(d_cur));
    checkCudaCall(cudaFree(d_next));

    /* Return pointer to array with final results */
    return current_array;
}
