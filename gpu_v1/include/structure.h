#pragma once

typedef struct data_sample
{
    double* data;
    double* label;
    int sample_width;
    int sample_height;
    int sample_count;

    double* data_dev;
    double* label_dev;
} Sample;

typedef struct convolution_kernel
{
    double* weight;
    double* delta_weight;

    double* weight_dev;
    double* delta_weight_dev;
}Kernel;

typedef struct layer_map
{
    double* data;
    double* error;
    double bias;
    double delta_bias;

    double* data_dev;
    double* error_dev;
}Map;

typedef struct layer
{
    int map_width;
    int map_height;
    int map_count;
    Map* map;

    int kernel_width;
    int kernel_height;
    int kernel_count;
    Kernel* kernel;
    double* map_common;
    
    double* map_common_dev;
}Layer;