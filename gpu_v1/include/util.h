#pragma once
#include "structure.h"
#include <iostream>
#include <fstream>
#include <cuda_runtime_api.h>

using namespace std;

// 保存 Layer 数据到文件
void save_layer(const Layer& layer, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    out.write(reinterpret_cast<const char*>(&layer.map_width), sizeof(layer.map_width));
    out.write(reinterpret_cast<const char*>(&layer.map_height), sizeof(layer.map_height));
    out.write(reinterpret_cast<const char*>(&layer.map_count), sizeof(layer.map_count));

    for (int i = 0; i < layer.map_count; ++i) {
        int map_size = layer.map_width * layer.map_height;
        out.write(reinterpret_cast<const char*>(layer.map[i].data), map_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(layer.map[i].error), map_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(&layer.map[i].bias), sizeof(double));
        out.write(reinterpret_cast<const char*>(&layer.map[i].delta_bias), sizeof(double));
    }

    out.write(reinterpret_cast<const char*>(&layer.kernel_width), sizeof(layer.kernel_width));
    out.write(reinterpret_cast<const char*>(&layer.kernel_height), sizeof(layer.kernel_height));
    out.write(reinterpret_cast<const char*>(&layer.kernel_count), sizeof(layer.kernel_count));

    for (int i = 0; i < layer.kernel_count; ++i) {
        int kernel_size = layer.kernel_width * layer.kernel_height;
        out.write(reinterpret_cast<const char*>(layer.kernel[i].weight), kernel_size * sizeof(double));
        out.write(reinterpret_cast<const char*>(layer.kernel[i].delta_weight), kernel_size * sizeof(double));
    }

    int common_size = layer.map_width * layer.map_height;
    out.write(reinterpret_cast<const char*>(layer.map_common), common_size * sizeof(double));

    out.close();
}

// 从文件读取 Layer 数据
void load_layer(Layer* layer, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    in.read(reinterpret_cast<char*>(&layer->map_width), sizeof(layer->map_width));
    in.read(reinterpret_cast<char*>(&layer->map_height), sizeof(layer->map_height));
    in.read(reinterpret_cast<char*>(&layer->map_count), sizeof(layer->map_count));

    layer->map = new Map[layer->map_count];
    int map_size = layer->map_width * layer->map_height;
    for (int i = 0; i < layer->map_count; ++i) {
        layer->map[i].data = new double[map_size];
        layer->map[i].error = new double[map_size];
        in.read(reinterpret_cast<char*>(layer->map[i].data), map_size * sizeof(double));
        in.read(reinterpret_cast<char*>(layer->map[i].error), map_size * sizeof(double));
        in.read(reinterpret_cast<char*>(&layer->map[i].bias), sizeof(double));
        in.read(reinterpret_cast<char*>(&layer->map[i].delta_bias), sizeof(double));
    }

    in.read(reinterpret_cast<char*>(&layer->kernel_width), sizeof(layer->kernel_width));
    in.read(reinterpret_cast<char*>(&layer->kernel_height), sizeof(layer->kernel_height));
    in.read(reinterpret_cast<char*>(&layer->kernel_count), sizeof(layer->kernel_count));

    layer->kernel = new Kernel[layer->kernel_count];
    int kernel_size = layer->kernel_width * layer->kernel_height;
    for (int i = 0; i < layer->kernel_count; ++i) {
        layer->kernel[i].weight = new double[kernel_size];
        layer->kernel[i].delta_weight = new double[kernel_size];
        in.read(reinterpret_cast<char*>(layer->kernel[i].weight), kernel_size * sizeof(double));
        in.read(reinterpret_cast<char*>(layer->kernel[i].delta_weight), kernel_size * sizeof(double));
    }

    int common_size = layer->map_width * layer->map_height;
    layer->map_common = new double[common_size];
    in.read(reinterpret_cast<char*>(layer->map_common), common_size * sizeof(double));

    in.close();
}

// 从设备传输数据到主机
void device_to_host_transfer(double* deviceData, double* hostData, size_t dataSize) {
    cudaMemcpy(hostData, deviceData, dataSize * sizeof(double), cudaMemcpyDeviceToHost);
}

// 从主机传输数据到设备
void host_to_device_transfer(double* hostData, double* deviceData, size_t dataSize) {
    cudaMemcpy(deviceData, hostData, dataSize * sizeof(double), cudaMemcpyHostToDevice);
}

void release_layer(Layer* layer)
{

    for (int i = 0; i < layer->kernel_count; i++)
    {
        free(layer->kernel[i].weight);
        free(layer->kernel[i].delta_weight);
        layer->kernel[i].weight = NULL;
        layer->kernel[i].delta_weight = NULL;
    }
    free(layer->kernel);
    layer->kernel = NULL;

    for (int i = 0; i < layer->map_count; i++)
    {
        free(layer->map[i].data);
        free(layer->map[i].error);
        layer->map[i].data = NULL;
        layer->map[i].error = NULL;
    }
    free(layer->map_common);
    layer->map_common = NULL;
    free(layer->map);
    layer->map = NULL;
}


void copy_to_device(Layer* layer) {
    // 为每个 Kernel 分配设备内存并复制权重
    int kernel_size = layer->kernel_width * layer->kernel_height;
    for (int i = 0; i < layer->kernel_count; ++i) {
        cudaMalloc(&layer->kernel[i].weight_dev, kernel_size * sizeof(double));
        cudaMemcpy(layer->kernel[i].weight_dev, layer->kernel[i].weight, kernel_size * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&layer->kernel[i].delta_weight_dev, kernel_size * sizeof(double));
        cudaMemcpy(layer->kernel[i].delta_weight_dev, layer->kernel[i].delta_weight, kernel_size * sizeof(double), cudaMemcpyHostToDevice);
    }

    // 为每个 Map 分配设备内存
    int map_size = layer->map_width * layer->map_height;
    for (int i = 0; i < layer->map_count; ++i) {
        cudaMalloc(&layer->map[i].data_dev, map_size * sizeof(double));
        cudaMemcpy(layer->map[i].data_dev, layer->map[i].data, map_size * sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&layer->map[i].error_dev, map_size * sizeof(double));
        cudaMemcpy(layer->map[i].error_dev, layer->map[i].error, map_size * sizeof(double), cudaMemcpyHostToDevice);
    }

    // 为 map_common_dev 分配设备内存
    cudaMalloc(&layer->map_common_dev, map_size * sizeof(double));
    cudaMemcpy(layer->map_common_dev, layer->map_common, map_size * sizeof(double), cudaMemcpyHostToDevice);
}

void free_layer_device_memory(Layer* layer) {
    // 释放 Kernel 的设备内存
    for (int i = 0; i < layer->kernel_count; ++i) {
        cudaFree(layer->kernel[i].weight_dev);
        cudaFree(layer->kernel[i].delta_weight_dev);
    }

    // 释放 Map 的设备内存
    for (int i = 0; i < layer->map_count; ++i) {
        cudaFree(layer->map[i].data_dev);
        cudaFree(layer->map[i].error_dev);
    }

    // 释放 map_common_dev 的设备内存
    cudaFree(layer->map_common_dev);
}
