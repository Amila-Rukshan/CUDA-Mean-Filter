#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <tuple>
#include <iostream>

// round double to int
int my_round(double d)
{
    int y = (int)(d + 0.5+(d<0));
    return y;
}

void meanFilter_h(unsigned char* raw_image_matrix, int image_width, int image_height, int window_size)
{
    int size = 3 * image_width * image_height;
    unsigned char* filtered_image_data = new unsigned char[size];
    int half_window = (window_size-window_size % 2)/2;
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
}

__global__ void meanFilter_d(unsigned char* raw_image_matrix, int image_width, int image_height, int window_size)
{
   
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
    printf("Image dimensions: %d * %d\n",width,height);
    printf("     Window size: %d\n",window_size);

    int size = 3 * width * abs(height);
    unsigned char* data = new unsigned char[size]; // allocate 3 bytes per pixel
    fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
    fclose(f);

    //******reading the bitmap to char array******
    
    // for(int i = 0; i < size; i += 1)
    // {
    //     printf("%d\n",data[i]);
    // }
    
    // call to CPU code
    clock_t start_h = clock();
    printf("Doing CPU Mean Filter...\n");
    meanFilter_h(data, width, height, window_size);
    clock_t end_h = clock();
    double time_h = (double)(end_h-start_h)/CLOCKS_PER_SEC;
    
    printf("CPU Time: %f\n",time_h);

    // call to GPU code
    return 0;
}

