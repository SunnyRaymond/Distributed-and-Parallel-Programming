/*
 * checksum.cu
 *
 * You can implement the CUDA-accelerated checksum calculator in this file.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#include "timer.hh"
#include "file.hh"

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

/* Change this kernel to compute a simple, additive checksum of the given data.
 * The result should be written to the given result-integer, which is an
 * integer and NOT an array like deviceDataIn. */
 __global__ void checksumKernel(unsigned int* result, unsigned int *deviceDataIn){

    // YOUR CODE HERE

}

/* Wrapper for your checksum kernel, i.e., does the necessary preparations and
 * calls your kernel. */
unsigned int checksumSeq (int n, unsigned int* data_in) {
    int i;
    timer sequentialTime = timer("Sequential checksum");

    sequentialTime.start();
    for (i=0; i<n; i++) {}
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Checksum (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}

/**
 * The checksumCuda handler that initialises the arrays to be used and calls
 * the checksum kernel. It also computes the missing values not calculated
 * on the GPU. It then adds all values together and prints the checksum
 */
 unsigned int checksumCuda (int n, unsigned int* data_in) {
    int threadBlockSize = 512;

    // Allocate the vectors & the result int on the GPU
    unsigned int* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(unsigned int)));
    if (deviceDataIn == NULL) {
        cout << "Could not allocate input data on GPU." << endl;
        exit(1);
    }
    unsigned int* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, sizeof(unsigned int)));
    if (deviceResult == NULL) {
        cout << "Could not allocate result integer on GPU." << endl;
        exit(1);
    }

    timer kernelTime  = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // Copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(unsigned int), cudaMemcpyHostToDevice));
    memoryTime.stop();

    kernelTime.start();
    checksumKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceResult, deviceDataIn);
    cudaDeviceSynchronize();
    kernelTime.stop();

    // Check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // Copies back the correct data
    unsigned int result;
    checkCudaCall(cudaMemcpy(&result, deviceResult, sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // Releases the GPU data
    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceResult));

    // The times are printed
    cout << fixed << setprecision(6);
    cout << "Kernel: \t\t" << kernelTime.getElapsed() << " seconds." << endl;
    cout << "Memory: \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return result;
}

/* Entry point to the program. */
int main(int argc, char* argv[]) {
    int n;
    char* mode;
    char* fileName;

    // Arg parse
    if (argc == 3) {
        fileName = argv[1];
        mode = argv[2];

        cout << "Running in '" << mode << "' mode" << endl;
        cout << "Opening file " << fileName << endl;
    } else {
        cout << "Usage: " << argv[0] << " filename mode" << endl;
        cout << " - filename: name of the file for which the checksum will be "
                "computed." << endl;
        cout << " - mode: one of the three modes for which the program can "
                "run." << endl;
        cout << "   Available options are:" << endl;
        cout << "    * seq: only runs the sequential implementation" << endl;
        cout << "    * cuda: only runs the parallelized implementation" << endl;
        cout << "    * both: runs both the sequential and the parallelized "
                "implementation" << endl;

        return EXIT_FAILURE;
    }
    n = fileSize(fileName);
    if (n == -1) {
        cerr << "File '" << fileName << "' not found" << endl;
        exit(EXIT_FAILURE);
    }

    char* data_in = new char[n];
    readData(fileName, data_in);
    unsigned int *data_in_raw = new unsigned int[n];
    for (int i = 0; i < n; i++){
        data_in_raw[i] = data_in[i];
    }

    /* Check the option to determine the functions to be called */
    if (strcmp(mode, "seq") == 0){
        // Only sequential checkusm is ran
        unsigned int checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else if (strcmp(mode, "cuda") == 0) {
        // Only cuda checksum is ran
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
    } else if (strcmp(mode, "both") == 0){
        // Both the sequential and the cuda checksum are run
        unsigned int checksum = checksumCuda(n, data_in_raw);
        cout << "CUDA checksum: " << checksum << endl;
        checksum = checksumSeq(n, data_in_raw);
        cout << "Sequential checksum: " << checksum << endl;
    } else {
        cerr << "Unknown mode '" << mode << "'; only accepts 'seq', 'cuda' or "
                "'both'" << endl;
        delete[] data_in;
        delete[] data_in_raw;
        exit(EXIT_FAILURE);
    }

    delete[] data_in;
    delete[] data_in_raw;
    return EXIT_SUCCESS;
}
