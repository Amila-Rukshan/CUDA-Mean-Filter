#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <tuple>
#include <iostream>
#include <string.h>
#include <chrono>

/*
 * compile: nvcc meanFilter.cu -o meanFilter
 * run: ./meanFilter <string: image.bmp path> <int: size of the square filter>
 */

long time_h = 0;
long time_d = 0;

int sample_rounds = 10;

// round double to int
int my_round(double d)
{
    int y = (int)(d + 0.5+(d<0));
    return y;
}

void meanFilter_h(unsigned char* raw_image_matrix,unsigned char* filtered_image_data,int image_width, int image_height, int half_window)
{
    
    for(int i = 0; i < image_height; i += 1){
        for(int j = 0; j < image_width; j += 1){
            int k = 3*(i*image_height+j);
            int top, bottom, left, right; 
            if(i-half_window >= 0){top = i-half_window;}else{top = 0;}// top limit
            if(i+half_window <= image_height-1){bottom = i+half_window;}else{bottom = image_height-1;}// bottom limit
            if(j-half_window >= 0){left = j-half_window;}else{left = 0;}// left limit
            if(j+half_window <= image_width-1){right = j+half_window;}else{right = image_width-1;}// right limit
            double first_byte = 0; 
            double second_byte = 0; 
            double third_byte = 0; 
            // move inside the window
            for(int x = top; x <= bottom; x++){
                for(int y = left; y <= right; y++){
                    int pos = 3*(x*image_height + y); // three bytes
                    first_byte += raw_image_matrix[pos];
                    second_byte += raw_image_matrix[pos+1];
                    third_byte += raw_image_matrix[pos+2];
                }
            }
            int effective_window_size = (bottom-top+1)*(right-left+1);
            filtered_image_data[k] = first_byte/effective_window_size;
            filtered_image_data[k+1] = second_byte/effective_window_size;
            filtered_image_data[k+2] =third_byte/effective_window_size;

            
        }
    }
    // printf("Result from CPU\n");
    // for(int z = 0; z < size; z += 3)
    // {
    //     printf("(%d, %d, %d)\n",filtered_image_data[z], filtered_image_data[z+1], filtered_image_data[z+2]);
    // }
}

__global__ void meanFilter_d(unsigned char* raw_image_matrix, unsigned char* filtered_image_data, int image_width, int image_height, int half_window)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < image_height && j < image_width){
        int k = 3*(i*image_height+j);
        int top, bottom, left, right; 
        if(i-half_window >= 0){top = i-half_window;}else{top = 0;}// top limit
        if(i+half_window <= image_height-1){bottom = i+half_window;}else{bottom = image_height-1;}// bottom limit
        if(j-half_window >= 0){left = j-half_window;}else{left = 0;}// left limit
        if(j+half_window <= image_width-1){right = j+half_window;}else{right = image_width-1;}// right limit
        double first_byte = 0; 
        double second_byte = 0; 
        double third_byte = 0; 
        // move inside the window
        for(int x = top; x <= bottom; x++){
            for(int y = left; y <= right; y++){
                int pos = 3*(x*image_height + y); // three bytes
                first_byte += raw_image_matrix[pos];
                second_byte += raw_image_matrix[pos+1];
                third_byte += raw_image_matrix[pos+2];
            }
        }
        int effective_window_size = (bottom-top+1)*(right-left+1);
        filtered_image_data[k] = first_byte/effective_window_size;
        filtered_image_data[k+1] = second_byte/effective_window_size;
        filtered_image_data[k+2] =third_byte/effective_window_size;
    }
}


int main(int argc,char **argv)
{
    printf("Starting...\n");
    
    //******reading the bitmap to char array******
    FILE* f = fopen(argv[1], "rb");
    unsigned char info[54];
    fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

    // extract image height and width from header
    int width, height;
    memcpy(&width, info + 18, sizeof(int));
    memcpy(&height, info + 22, sizeof(int));

    int window_size = strtol(argv[2],NULL,10);
    printf("     Window size: %d\n",window_size);
    printf("Image dimensions: (%d, %d)\n",width,height);
        
    int size = 3 * width * abs(height);
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    unsigned char* result_image_data_d;
    unsigned char* result_image_data_h = new unsigned char[size];
    unsigned char* result_image_data_h1 = new unsigned char[size];

    unsigned char* image_data_d;

    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);
    //******reading the bitmap to char array******

    // dim3 dimGrid (GRID_SIZE, GRID_SIZE);
    // dim3 dimBlock (BLOCK_SIZE, BLOCK_SIZE);
    int block_size = 32;
    int grid_size = width/block_size;
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(grid_size, grid_size, 1);

    printf("       GRID_SIZE: (%d, %d)\n", grid_size, grid_size);
    printf("      BLOCK_SIZE: (%d, %d)\n", block_size, block_size);
    
    for(int _ = 0; _ < sample_rounds; _ += 1)
    {
        printf("Allocating device memory on host..\n");
        cudaMalloc((void **)&image_data_d,size*sizeof(unsigned char));
        cudaMalloc((void **)&result_image_data_d,size*sizeof(unsigned char));
        printf("Copying to device..\n");
        cudaMemcpy(image_data_d,data,size*sizeof(unsigned char),cudaMemcpyHostToDevice);
        int half_window = (window_size-window_size % 2)/2;
        printf("half window: %d\n", half_window);

        // call to GPU code
        std::chrono::steady_clock::time_point start_d = std::chrono::high_resolution_clock::now();
        printf("Doing GPU Mean Filter...\n");
        meanFilter_d <<< dimGrid, dimBlock >>> (image_data_d, result_image_data_d, width, height, half_window);
        cudaThreadSynchronize();

        cudaError_t error = cudaGetLastError();
        if(error!=cudaSuccess)
        {
            fprintf(stderr,"ERROR: %s\n", cudaGetErrorString(error) );
            exit(-1);
        }
        std::chrono::steady_clock::time_point end_d = std::chrono::high_resolution_clock::now();

        // call to CPU code
        std::chrono::steady_clock::time_point start_h = std::chrono::high_resolution_clock::now();
        printf("Doing CPU Mean Filter...\n");
        meanFilter_h(data, result_image_data_h1, width, height, half_window);
        std::chrono::steady_clock::time_point end_h = std::chrono::high_resolution_clock::now();

        printf("Result from GPU\n");
        cudaMemcpy(result_image_data_h,result_image_data_d,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

        printf("compare results code : %d\n",memcmp(result_image_data_h, result_image_data_h1, size*sizeof(unsigned char)));

        time_h += std::chrono::duration_cast<std::chrono::microseconds>(end_h-start_h).count(); 
        time_d += std::chrono::duration_cast<std::chrono::microseconds>(end_d-start_d).count();

        cudaFree(image_data_d);
        cudaFree(result_image_data_d);
    }
    // cudaMemcpy(data_h,image_data_d,size*sizeof(unsigned char),cudaMemcpyDeviceToHost);

    // for(int i = 0; i < size; i += 3)
    // {
    //     printf("(%d, %d, %d)\n",result_image_data_h[i], result_image_data_h[i+1], result_image_data_h[i+2]);
    // }

    printf("    GPU Time: %d microseconds\n",(time_d/sample_rounds));
    printf("    CPU Time: %d microseconds\n",(time_h/sample_rounds));
    printf("CPU/GPU time: %f \n",((float)time_h/time_d));

    
    return 0;
}

