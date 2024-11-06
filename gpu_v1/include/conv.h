#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 朴素的CPU卷积计算
void convolution_calcu(double* input_map_data, int input_map_width, int input_map_height, double* kernel_data, int kernel_width, int kernel_height, double* result_map_data, int result_map_width, int result_map_height)
{

    double sum = 0.0;
    for (int i = 0; i < result_map_height; i++)
    {
        for (int j = 0; j < result_map_width; j++)
        {
            sum = 0.0;
            for (int n = 0; n < kernel_height; n++)
            {
                for (int m = 0; m < kernel_width; m++)
                {
                    int index_input_reshuffle = (i + n) * input_map_width + j + m;
                    int index_kernel_reshuffle = n * kernel_width + m;
                    sum += input_map_data[index_input_reshuffle] * kernel_data[index_kernel_reshuffle];
                }
            }
            int index_result_reshuffle = i * result_map_width + j;
            result_map_data[index_result_reshuffle] += sum;
        }
    }
}

// 朴素的GPU卷积计算
__global__ void conv2D(double* input, int input_width, int input_height,
                       double* kernel, int kernel_width, int kernel_height,
                       double* output, int output_width, int output_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < output_width && y < output_height) {
        double sum = 0.0;
        for (int ky = 0; ky < kernel_height; ++ky) {
            for (int kx = 0; kx < kernel_width; ++kx) {
                int ix = x + kx;
                int iy = y + ky;
                if (ix < input_width && iy < input_height) {
                    sum += input[iy * input_width + ix] * kernel[ky * kernel_width + kx];
                }
            }
        }
        output[y * output_width + x] = sum;
    }
}

void runConv2D(double* d_input, int input_width, int input_height,
               double* d_kernel, int kernel_width, int kernel_height,
               double* d_output, int output_width, int output_height) {
    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((output_width + blockSize.x - 1) / blockSize.x,
                  (output_height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    conv2D<<<gridSize, blockSize>>>(d_input, input_width, input_height,
                                    d_kernel, kernel_width, kernel_height,
                                    d_output, output_width, output_height);
}