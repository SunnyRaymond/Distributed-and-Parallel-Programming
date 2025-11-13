/*
 * vector-add.cu
 *
 * Performs a vector addition on the GPU. You can use this as an example for an
 * application parallelized with CUDA.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "timer.hh"
#include <iostream>

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
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


/* The kernel, which runs on the GPU. */
__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceResult) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    deviceResult[index] = deviceA[index] + deviceB[index];
}


/* Function that prepares & copies data to the GPU, runs the kernel, and then
 * copies the result. back. */
void vectorAddCuda(int n, float* a, float* b, float* result) {
    int threadBlockSize = 512;

    // Allocate the vectors on the GPU, each time checking if we were successfull
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        cerr << "Could not allocate A array on GPU." << endl;
        return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cerr << "Could not allocate B array on GPU." << endl;
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cerr << "Could not allocate Result array on GPU." << endl;
        return;
    }

    // Prepare the CUDA timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    // Execute the vector-add kernel
    cudaEventRecord(start, 0);
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    cudaEventRecord(stop, 0);

    // Check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // Copy result back to host
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup GPU-side data
    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    // Print the time the kernel invocation took, without the copies!
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    cout << "Kernel invocation took " << elapsedTime << " milliseconds" << endl;
}


/* Entry point for the vector add. */
int main(int argc, char* argv[]) {
    int n = 65536;
    timer vectorAddTimer("vector add timer");
    float* a = new float[n];
    float* b = new float[n];
    float* result = new float[n];

    // Initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Run the vector add, and also time including the copies
    vectorAddTimer.start();
    vectorAddCuda(n, a, b, result);
    vectorAddTimer.stop();

    cout << vectorAddTimer;

    // verify the resuls
    for(int i=0; i<n; i++) {
        if(result[i] != 2*i) {
            cerr << "Error in results! Element " << i << " is " << result[i] << ", but should be " << (2*i) << endl;
            exit(1);
        }
    }
    cout << "results OK!" << endl;
    
    // Cleanup the CPU-side data
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}
