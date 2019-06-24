#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <string.h>

/*
 * compile: nvcc .\vecadd.cu -o vecadd 
 * run: ./vecadd <int: size of the vector> <int: block size>
 */

int *a, *b;  // host data
int *c, *c2;  // results

int sample_size = 10;
double time_d = 0;
double time_h = 0;

int n; // size of the vector

__global__ void vecAdd(int *A,int *B,int *C,int N)
{
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < N)
      C[i] = A[i] + B[i];
}

void vecAdd_h(int *A1,int *B1, int *C1, int N)
{
   for(int i=0;i<N;i++)
      C1[i] = A1[i] + B1[i];
}

int main(int argc,char **argv)
{
   printf("Begin \n");
   n = strtol(argv[1], NULL, 10);
   int nBytes = n*sizeof(int);
   int block_size, block_no;
   a = (int *)malloc(nBytes);
   b = (int *)malloc(nBytes);
   c = (int *)malloc(nBytes);
   c2 = (int *)malloc(nBytes);
   int *a_d,*b_d,*c_d;
   block_size = strtol(argv[2], NULL, 10);;
   block_no = ceil(n/block_size);
   dim3 dimBlock(block_size,1,1);
   dim3 dimGrid(block_no,1,1);

   for(int i = 0; i < n; i++ ) {
        a[i] = 1;
        b[i] = 1;
   }

   for(int _ = 0; _ < sample_size; _ += 1)
   {

      printf("Allocating device memory on host..\n");
      cudaMalloc((void **)&a_d,n*sizeof(int));
      cudaMalloc((void **)&b_d,n*sizeof(int));
      cudaMalloc((void **)&c_d,n*sizeof(int));
      printf("Copying to device..\n");
      cudaMemcpy(a_d,a,n*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(b_d,b,n*sizeof(int),cudaMemcpyHostToDevice);
      clock_t start_d=clock();
      printf("Doing GPU Vector add\n");
      vecAdd<<<block_no,block_size>>>(a_d,b_d,c_d,n);
      cudaThreadSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error!=cudaSuccess)
      {
         fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
         exit(-1);
      }
      clock_t end_d = clock();
      clock_t start_h = clock();
      printf("Doing CPU Vector add\n");
      vecAdd_h(a,b,c2,n);
      clock_t end_h = clock();
      time_d += (double)(end_d-start_d)/CLOCKS_PER_SEC;
      time_h += (double)(end_h-start_h)/CLOCKS_PER_SEC;
      cudaMemcpy(c,c_d,n*sizeof(int),cudaMemcpyDeviceToHost);
      // for(int i = 0; i < n; i += 1)
      // {
      //    printf("%d : %d\n",i, c[i]);
      // }

      printf("compare results code : %d\n",memcmp(c, c2, n*sizeof(int)));
      
      cudaFree(a_d);
      cudaFree(b_d);
      cudaFree(c_d);
   }

   printf("Number of elements: %d GPU Time: %f CPU Time: %f\n", n, time_d/sample_size, time_h/sample_size);
   
   return 0;
}