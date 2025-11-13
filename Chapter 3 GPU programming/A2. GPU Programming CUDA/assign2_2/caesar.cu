/*
 * caesar.cu
 *
 * You can implement your CUDA-accelerated encryption and decryption algorithms
 * in this file.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#include "file.hh"
#include "timer.hh"

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

/* Change this kernel to properly encrypt the given data. The result should be
 * written to the given out data. */
__global__ void encryptKernel(char* deviceDataIn, char* deviceDataOut,
                              int n, int key_length, const int *key) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Shift cipher: c[i] = (byte)(p[i] + key[i mod key_length])
    int k = key[idx % key_length];
    unsigned char in = static_cast<unsigned char>(deviceDataIn[idx]);
    unsigned char out = static_cast<unsigned char>(in + k); // wraps mod 256
    deviceDataOut[idx] = static_cast<char>(out);
}

/* Change this kernel to properly decrypt the given data. The result should be
 * written to the given out data. */
__global__ void decryptKernel(char* deviceDataIn, char* deviceDataOut,
                              int n, int key_length, const int *key) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Inverse shift: p[i] = (byte)(c[i] - key[i mod key_length])
    int k = key[idx % key_length];
    unsigned char in = static_cast<unsigned char>(deviceDataIn[idx]);
    unsigned char out = static_cast<unsigned char>(in - k); // wraps mod 256
    deviceDataOut[idx] = static_cast<char>(out);
}

/* Sequential implementation of encryption with the Shift cipher (and therefore
 * also of Caesar's cipher, if key_length == 1), which you need to implement as
 * well. Then, it can be used to verify your parallel results and compute
 * speedups of your parallelized implementation. */
int EncryptSeq (int n, char* data_in, char* data_out, int key_length, int *key)
{
  int i;
  timer sequentialTime = timer("Sequential encryption");

  sequentialTime.start();
  for (i = 0; i < n; i++) {

    // YOUR CODE HERE
    int k = key[i % key_length];
    unsigned char in = static_cast<unsigned char>(data_in[i]);
    unsigned char out = static_cast<unsigned char>(in + k); // modulo-256
    data_out[i] = static_cast<char>(out);

  }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

  return 0;
}

/* Sequential implementation of decryption with the Shift cipher (and therefore
 * also of Caesar's cipher, if key_length == 1), which you need to implement as
 * well. Then, it can be used to verify your parallel results and compute
 * speedups of your parallelized implementation. */
int DecryptSeq (int n, char* data_in, char* data_out, int key_length, int *key)
{
  int i;
  timer sequentialTime = timer("Sequential decryption");

  sequentialTime.start();
  for (i = 0; i < n; i++) {

    // YOUR CODE HERE
    int k = key[i % key_length];
    unsigned char in = static_cast<unsigned char>(data_in[i]);
    unsigned char out = static_cast<unsigned char>(in - k); // modulo-256
    data_out[i] = static_cast<char>(out);

  }
  sequentialTime.stop();

  cout << fixed << setprecision(6);
  cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

  return 0;
}

/* Wrapper for your encrypt kernel, i.e., does the necessary preparations and
 * calls your kernel. */
int EncryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;

    if (n <= 0) {
        return 0;
    }

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    // copy key to device
    int *deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **)&deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        checkCudaCall(cudaFree(deviceDataOut));
        cout << "could not allocate key memory!" << endl;
        return -1;
    }
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length * sizeof(int),
                             cudaMemcpyHostToDevice));

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    int gridSize = (n + threadBlockSize - 1) / threadBlockSize;

    kernelTime1.start();
    encryptKernel<<<gridSize, threadBlockSize>>>(deviceDataIn, deviceDataOut,
                                                 n, key_length, deviceKey);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

/* Wrapper for your decrypt kernel, i.e., does the necessary preparations and
 * calls your kernel. */
int DecryptCuda (int n, char* data_in, char* data_out, int key_length, int *key) {
    int threadBlockSize = 512;

    if (n <= 0) {
        return 0;
    }

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    // copy key to device
    int *deviceKey = NULL;
    checkCudaCall(cudaMalloc((void **)&deviceKey, key_length * sizeof(int)));
    if (deviceKey == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        checkCudaCall(cudaFree(deviceDataOut));
        cout << "could not allocate key memory!" << endl;
        return -1;
    }
    checkCudaCall(cudaMemcpy(deviceKey, key, key_length * sizeof(int),
                             cudaMemcpyHostToDevice));

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    int gridSize = (n + threadBlockSize - 1) / threadBlockSize;

    kernelTime1.start();
    decryptKernel<<<gridSize, threadBlockSize>>>(deviceDataIn, deviceDataOut,
                                                 n, key_length, deviceKey);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceKey));

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

/* Entry point to the function! */
int main(int argc, char* argv[]) {
    // Check if there are enough arguments
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " key..." << endl;
        cout << " - key: one or more values for the encryption key, separated "
                "by spaces" << endl;
        
        return EXIT_FAILURE;
    }

    // Parse the keys from the command line arguments
    int key_length = argc - 1;
    int *enc_key = new int[key_length];
    for (int i = 0; i < key_length; i++) {
        enc_key[i] = atoi(argv[i + 1]);
    }

    // Check if the original.data file exists and what it's size is
    int n;
    n = fileSize("original.data");
    if (n == -1) {
        cout << "File not found! Exiting ... " << endl;
        exit(0);
    }

    // Read the file in memory from the disk
    char* data_in = new char[n];
    char* data_out = new char[n];
    readData("original.data", data_in);

    cout << "Encrypting a file of " << n << " characters." << endl;

    EncryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential.data", data_out);
    EncryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "cuda.data", data_out);

    readData("cuda.data", data_in);

    cout << "Decrypting a file of " << n << "characters" << endl;
    DecryptSeq(n, data_in, data_out, key_length, enc_key);
    writeData(n, "sequential_recovered.data", data_out);
    DecryptCuda(n, data_in, data_out, key_length, enc_key);
    writeData(n, "recovered.data", data_out);

    delete[] data_in;
    delete[] data_out;

    return 0;
}
