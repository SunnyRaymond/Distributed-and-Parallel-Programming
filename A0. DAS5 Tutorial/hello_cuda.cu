#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

#define SAMPLES 10

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("CUDA error: %s \n",  cudaGetErrorString(result));
        exit(1);
    }
}


__global__ void helloKernel(int n, int* A) {
// insert operation here
   size_t id = blockIdx.x * blockDim.x + threadIdx.x;
   if (id < n) A[id]=id; 
}

void helloCuda(int n, int* a) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    int* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(int)));
    if (deviceA == NULL) {
        printf("Could not allocate memory for A. \n");
        return;
    }
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(int), cudaMemcpyHostToDevice));
    helloKernel<<<n/threadBlockSize, threadBlockSize>>>(n,deviceA);
    cudaDeviceSynchronize();
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());
    // copy result back
    checkCudaCall(cudaMemcpy(a, deviceA, n * sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
}

int main(int argc, char* argv[]) {
    int n = 655360;
    int* a = new int[n];

    if (argc > 1) n = atoi(argv[1]);

    printf("Testing CUDA! \n");
    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = 0;
    }

    helloCuda(n, a);
   
    srand(n); 
    // verify the resuls
    for(int i=0; i<SAMPLES; i++) {
	  int j = rand() % n; 
	  if (j!=a[j]) {
            printf("Error in results! Element %d is %d, but should be %d! \n", j, j, a[j]); 
            exit(1);
        }
    }
    printf("results OK! \n");
            
    delete[] a;
    
    return 0;
}
