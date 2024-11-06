#include "util.h"
#include "structure.h"
#include "conv.h"
#include <ctime>
#include <memory.h>
#include <cstdlib>
#include <cmath>

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),  (mode)))==NULL
#endif

#ifndef MAX
#define MAX(A,B)	(((A) > (B)) ? (A) : (B))
#endif

using namespace std;

const int batch_size = 10;
const int class_count = 10;
const int width = 32;
const int height = 32;
const int train_sample_count = 60000;
const int test_sample_count = 10000;
const int zero_padding = 2;
const int padded_matrix_size = (28 + zero_padding * 2) * (28 + 2 * zero_padding);

Layer input_layer;
Layer output_layer;
Layer c1_convolution_layer;
Layer s2_pooling_layer;
Layer c3_convolution_layer;
Layer s4_pooling_layer;
Layer c5_convolution_layer;

bool is_logging = false;
void log_exec(const char* msg)
{
    if (is_logging)
    {
        cout << "exec part in -->> [ " << msg << " ]" << endl;
    }
}

void show_pic(Sample sample, int target)
{
    cout << "img:" << endl;
    for (int w = 0; w < 1024; w++)
    {
        cout << sample.data[w] << ",";
        if ((w + 1) % 32 == 0)
        {
            cout << endl;
        }
    }
    cout << endl << "target:" << target << endl;
    cout << endl << endl;
}

void show_pic(Sample sample)
{
    cout << "img:" << endl;
    for (int w = 0; w < 1024; w++)
    {
        cout << sample.data[w] << ",";
        if ((w + 1) % 32 == 0)
        {
            cout << endl;
        }
    }
    int index = 0;
    for (int i = 0; i < class_count; i++)
    {
        if (sample.label[i] == 1)
        {
            index = i;
        }
    }
    cout << endl << "target:" << index << endl;
    cout << endl << endl;
}

void show_map(Map map, int width, int height)
{
    cout << "map info:" << endl;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            cout << map.data[i * width + j] << ",";
        }
        cout << endl;
    }
    cout << endl << endl;
}

void load_mnist_data(Sample* sample, const char* file_name, int sample_count) //checked 2.0
{
    FILE* mnist_file = NULL;
    int err = fopen_s(&mnist_file, file_name, "rb");

    unsigned char image_buffer[784];    //保存图片信息

    int head_info[1000];    //读取出文件的头信息，该信息不参与计算
    fread(head_info, 1, 16, mnist_file);   //读取16字节头部信息

    if (mnist_file == NULL)
    {
        cout << "load data from your file err..." << endl;
        return;
    }
    else
    {
        cout << "loading data...[in func -->> load_mnist_data]" << endl;
    }
    for (int i = 0; i < sample_count; i++)
    {
        sample[i].data = (double*)malloc(padded_matrix_size * sizeof(double));
        memset(sample[i].data, 0, padded_matrix_size * sizeof(double));
        fread(image_buffer, 1, 784, mnist_file);
        unsigned int value;
        int index = 0;
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 28; k++)
            {
                int shuffle_index = (j + zero_padding) * width + k + zero_padding;
                value = (unsigned int)image_buffer[index++];
                if (value < 128)
                {
                    sample[i].data[shuffle_index] = 0;
                }
                else
                {
                    sample[i].data[shuffle_index] = 1;
                }
            }
        }

    }
    fclose(mnist_file);
    mnist_file = NULL;
}

void load_mnist_label(Sample* sample, const char* file_name, const int sample_count) //checked 2.0
{
    FILE* mnist_file = NULL;
    int err = fopen_s(&mnist_file, file_name, "rb");
    unsigned char label;

    if (mnist_file == NULL)
    {
        cout << "load label from your file err..." << endl;
        return;
    }
    else
    {
        cout << "loading label...[in func -->> load_mnist_label]" << endl;
    }
    int head_info[1000];    //读取出文件的头信息，该信息不参与计算
    fread(head_info, 1, 8, mnist_file);    //读取8字节头部信息

    for (int i = 0; i < sample_count; i++)
    {
        sample[i].label = (double*)malloc(class_count * sizeof(double));
        for(int k = 0; k < class_count; k ++)
        {
            sample[i].label[k] = 0;
        }
        fread((char*)&label, sizeof(label), 1, mnist_file);
        int value_index = (unsigned int)label;
        sample[i].label[value_index] = 1;

        //show_pic(sample[i],value_index);  //test
    }
    fclose(mnist_file);
    mnist_file = NULL;
}
 
double weight_base_calc(int pre_map_count,int cur_map_count,int kernel_width,int kernel_height,bool is_pooling)
{
    const double scale = 6.0;
    int fan_in = 0;
    int fan_out = 0;
    if (is_pooling)
    {
        fan_in  = 4;
        fan_out = 1;
    }
    else
    {
        fan_in = pre_map_count * kernel_width * kernel_height;
        fan_out = cur_map_count * kernel_width * kernel_height;
    }
    int denominator = fan_in + fan_out;
    double weight_base = (denominator != 0) ? sqrt(scale / (double)denominator) : 0.5;
    return weight_base;
}
void init_kernel(double* kernel, int size, double weight_base)
{
    log_exec("init_kernel");

    for (int i = 0; i < size; i++)
    {
        kernel[i] = ((double)rand() / RAND_MAX - 0.5) * 2 * weight_base;
    }
}

void init_layer(Layer* layer, int prelayer_map_count, int map_count, int kernel_width, int kernel_height, int map_width, int map_height, bool is_pooling)
{
    log_exec("init_kernel");

    int single_kernel_memsize = 0;
    double weight_base = weight_base_calc(prelayer_map_count,map_count,kernel_width,kernel_height,is_pooling);

    //cout<<"weight base ："<<weight_base<<endl;

    layer->kernel_count = prelayer_map_count * map_count;//需要训练的kernel数目
    layer->kernel_width = kernel_width;
    layer->kernel_height = kernel_height;
    layer->kernel = (Kernel*)malloc(layer->kernel_count * sizeof(Kernel)); //为该层的所有卷积核分配内存
    single_kernel_memsize = layer->kernel_width * layer->kernel_height * sizeof(double);
    for (int i = 0; i < prelayer_map_count; i++) //初始化两层之间每一个卷积核的参数weight delta weight
    {
        for (int j = 0; j < map_count; j++)
        {
            int valued_index = (i * map_count) + j;
            layer->kernel[valued_index].weight = (double*)malloc(single_kernel_memsize);
            init_kernel(layer->kernel[valued_index].weight, kernel_height * kernel_width, weight_base);
            layer->kernel[valued_index].delta_weight = (double*)malloc(single_kernel_memsize);
            memset(layer->kernel[valued_index].delta_weight, 0, single_kernel_memsize);
        }
    }
    layer->map_count = map_count;
    layer->map_height = map_height;
    layer->map_width = map_width;
    layer->map = (Map*)malloc(layer->map_count * sizeof(Map));
    int single_map_size = layer->map_height * layer->map_width * sizeof(double);
    for (int i = 0; i < layer->map_count; i++) //初始化每一个map的参数 bias ,delta bias, data, error
    {
        layer->map[i].bias = 0.0;
        layer->map[i].delta_bias = 0.0;
        layer->map[i].data = (double*)malloc(single_map_size);
        memset(layer->map[i].data, 0, single_map_size);
        layer->map[i].error = (double*)malloc(single_map_size);
        memset(layer->map[i].error, 0, single_map_size);
    }
    layer->map_common = (double*)malloc(single_map_size);
    memset(layer->map_common, 0, single_map_size);
}

/*
    重置某layer层的所有map的delta_bias为0；
    重置某layer层所有的kernel的delta_weight为0
*/
void reset_layer_deltabias_kernel_deltaweight(Layer *layer) //checked 2.0
{

    log_exec("reset_layer_deltabias_kernel_deltaweight");

    int single_kernel_memsize = layer->kernel_height * layer->kernel_width * sizeof(double);
    for (int i = 0; i < layer->kernel_count; i++)
    {
        memset(layer->kernel[i].delta_weight, 0, single_kernel_memsize);
    }
    for (int i = 0; i < layer->map_count; i++)
    {
        layer->map[i].delta_bias = 0.0;
    }
}

void reset_all_layer() //checked 2.0
{
    log_exec("reset_all_layer");

    reset_layer_deltabias_kernel_deltaweight(&c1_convolution_layer);
    reset_layer_deltabias_kernel_deltaweight(&s2_pooling_layer);
    reset_layer_deltabias_kernel_deltaweight(&c3_convolution_layer);
    reset_layer_deltabias_kernel_deltaweight(&s4_pooling_layer);
    reset_layer_deltabias_kernel_deltaweight(&c5_convolution_layer);
    reset_layer_deltabias_kernel_deltaweight(&output_layer);
}
/*
    激活函数以及导函数计算
*/
struct activation_function  //checked 2.0
{
    inline static double tan_h(double x)
    {
        return ((exp(x) - exp(-x)) / (exp(x) + exp(-x)));
    }
    inline static double d_tan_h(double x)
    {
        return (1.0 - (x * x));
    }
    inline static double relu(double x)
    {
        return (x > 0.0 ? x : 0.0);
    }
    inline static double d_relu(double x)
    {
        return (x > 0.0 ? 1.0 : 0.0);
    }
    inline static double sigmod(double x)
    {
        return (1.0 / (1.0 + exp(-x)));
    }
    inline static double d_sigmod(double x)
    {
        return (x * (1.0 - x));
    }
};
/*
    损失函数
*/
struct loss_function //checked 2.0
{
    inline static double cal_loss(double real, double target)
    {
        return (real - target) * (real - target) / 2;
    }
    inline static double d_cal_loss(double real, double target)
    {
        return (real - target);
    }
};

/*
    单次计算map与kernel的卷积
*/


#define O true
#define X false
//S2到C3不完全连接映射表
bool connection_table[6 * 16] = {
                O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
                O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
                O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
                X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
                X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
                X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

void convolution_forward_propagation(Layer* pre_layer, Layer* cur_layer, bool* connection_table) //checked and fixed /checked 2.0
{
    log_exec("convolution_forward_propagation");
    int index_layer = 0;
    int layer_size = cur_layer->map_height * cur_layer->map_width;
    for (int i = 0; i < cur_layer->map_count; i++)
    {
        memset(cur_layer->map_common, 0, layer_size * sizeof(double)); //清空公共map的暂存数据。

        for (int j = 0; j < pre_layer->map_count; j++)
        {
            index_layer = j * cur_layer->map_count + i;
            if (connection_table != NULL && !connection_table[index_layer])
            {
                continue;
            }
            //fix para 3 map height
            runConv2D(pre_layer->map[j].data_dev, pre_layer->map_width, pre_layer->map_height,
                      cur_layer->kernel[index_layer].weight_dev, cur_layer->kernel_width, cur_layer->kernel_height,
                      cur_layer->map_common_dev, cur_layer->map_width, cur_layer->map_height);
            device_to_host_transfer(cur_layer->map_common_dev, cur_layer->map_common, sizeof(double) * layer_size);
        }
        for (int k = 0; k < layer_size; k++)
        {
            cur_layer->map[i].data[k] = activation_function::tan_h(cur_layer->map_common[k] + cur_layer->map[i].bias);
        }
        host_to_device_transfer(cur_layer->map[i].data, cur_layer->map[i].data_dev, sizeof(double) * layer_size);
    }
}
void max_pooling_forward_propagation(Layer* pre_layer, Layer* cur_layer) //checked 2.0
{
    log_exec("max_pooling_forward_propagation");

    int map_width = cur_layer->map_width;
    int map_height = cur_layer->map_height;
    int pre_map_width = pre_layer->map_width;

    for (int k = 0; k < cur_layer->map_count; k++)
    {
        for (int i = 0; i < map_height; i++)
        {
            for (int j = 0; j < map_width; j++)
            {
                double max_value = pre_layer->map[k].data[2 * i * pre_map_width + 2*j];
                for (int n = 2 * i; n < 2 * (i + 1); n++)
                {
                    for (int m = 2 * j; m < 2 * (j + 1); m++)
                    {
                        max_value = MAX(max_value, pre_layer->map[k].data[n * pre_map_width + m]);
                    }
                }
                cur_layer->map[k].data[i * map_width + j] = activation_function::tan_h(max_value);
            }
        }
    }
}

void fully_connection_forward_propagation(Layer* pre_layer, Layer* cur_layer)
{
    log_exec("fully_connection_forward_propagation");

    for (int i = 0; i < cur_layer->map_count; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < pre_layer->map_count; j++)
        {
            sum += pre_layer->map[j].data[0] * cur_layer->kernel[j * cur_layer->map_count + i].weight[0];
        }
        sum += cur_layer->map[i].bias;
        cur_layer->map[i].data[0] = activation_function::tan_h(sum);
    }
}

void forward_propagation()
{
    log_exec("forward_propagation");

    convolution_forward_propagation(&input_layer, &c1_convolution_layer, NULL);
    //show_map(c1_convolution_layer.map[0],28,28);
    max_pooling_forward_propagation(&c1_convolution_layer, &s2_pooling_layer);
    convolution_forward_propagation(&s2_pooling_layer, &c3_convolution_layer, connection_table);
    max_pooling_forward_propagation(&c3_convolution_layer, &s4_pooling_layer);
    convolution_forward_propagation(&s4_pooling_layer, &c5_convolution_layer, NULL);
    fully_connection_forward_propagation(&c5_convolution_layer, &output_layer);
}

void testing(Sample* samples)
{
    log_exec("testing");

    int success_count = 0;
    int result_test = 0;
    int result_label = 0;
    int single_sample_size = samples->sample_height * samples->sample_width * sizeof(double);
    int* result_test_label_matrix = (int*)malloc(class_count * class_count * sizeof(int));
    memset(result_test_label_matrix, 0, sizeof(int) * class_count * class_count);
    for (int i = 0; i < samples->sample_count; i++)
    {
        host_to_device_transfer(input_layer.map[0].data_dev, samples[i].data, single_sample_size);
        forward_propagation();

        int index_result = 0;
        double max_value = *(output_layer.map[0].data);
        for (int k = 0; k < output_layer.map_count; k++)
        {
            if (*(output_layer.map[k].data) > max_value)
            {
                max_value = *(output_layer.map[k].data);
                index_result = k;
            }
        }
        result_test = index_result;

        int index_label = 0;
        max_value = samples->label[0];
        for (int k = 1; k < class_count; k++)
        {
            if (samples[i].label[k] > max_value)
            {
                max_value = samples[i].label[k];
                index_label = k;
            }
        }
        result_label = index_label;

        if (result_test == result_label)
        {
            success_count++;
        }
        result_test_label_matrix[result_test * class_count + result_label] ++;
        if (i % 2500 == 0)
        {
            cout << "testing data process : " << i / (1.0 * samples->sample_count) * 100.0 << " %" << endl;
        }
    }

    //输出结果矩阵
    cout << "testing over !!! " << endl;
    cout << "success rate : " << (1.0) * success_count / samples->sample_count << endl;
    cout << "\t ";
    for (int i = 0; i < class_count; i++)
    {
        cout << "\t" << i;
    }
    cout << endl;
    for (int i = 0; i < class_count; i++)
    {
        cout << "\t" << i;
        for (int j = 0; j < class_count; j++)
        {
            cout << "\t" << result_test_label_matrix[i * class_count + j];
        }
        cout << endl;
    }


    int sum = 0;
    for (int i = 0; i < class_count * class_count; i++)
    {
        sum += result_test_label_matrix[i];
    }
    cout << "total sum : " << sum << endl;
    free(result_test_label_matrix);
    result_test_label_matrix = NULL;
}





int main()
{
    log_exec("MAIN");
    //初始化测试集样本
    auto* test_sample = static_cast<Sample *>(malloc(test_sample_count * sizeof(Sample)));
    memset(test_sample, 0, test_sample_count * sizeof(Sample));
    test_sample->sample_height = height;
    test_sample->sample_width = width;
    test_sample->sample_count = test_sample_count;
    const char* test_sample_path = "./../data/t10k-images.idx3-ubyte";
    const char* test_label_path = "./../data/t10k-labels.idx1-ubyte";
    load_mnist_data(test_sample, test_sample_path, 10000);
    load_mnist_label(test_sample, test_label_path, 10000);
    //初始化各层
    //init_layer(*layer, prelayer_map_count, map_count, kernel_width, kernel_height, map_width, map_height, is_pooling)
    init_layer(&input_layer, 0, 1, 0, 0, 32, 32, false);
    init_layer(&c1_convolution_layer, 1, 6, 5, 5, 28, 28, false);
    init_layer(&s2_pooling_layer, 1, 6, 1, 1, 14, 14, true);
    init_layer(&c3_convolution_layer, 6, 16, 5, 5, 10, 10, false);
    init_layer(&s4_pooling_layer, 1, 16, 1, 1, 5, 5, true);
    init_layer(&c5_convolution_layer, 16, 120, 5, 5, 1, 1, false);
    init_layer(&output_layer, 120, 10, 1, 1, 1, 1, false);
    // 加载权重
    load_layer(&input_layer, "./weight/input_layer");
    load_layer(&c1_convolution_layer, "./weight/c1_convolution_layer");
    load_layer(&s2_pooling_layer, "./weight/s2_pooling_layer");
    load_layer(&c3_convolution_layer, "./weight/c3_convolution_layer");
    load_layer(&s4_pooling_layer, "./weight/s4_pooling_layer");
    load_layer(&c5_convolution_layer, "./weight/c5_convolution_layer");
    load_layer(&output_layer, "./weight/output_layer");
    // 加载到device上
    copy_to_device(&input_layer);
    copy_to_device(&c1_convolution_layer);
    copy_to_device(&s2_pooling_layer);
    copy_to_device(&c3_convolution_layer);
    copy_to_device(&s4_pooling_layer);
    copy_to_device(&c5_convolution_layer);
    copy_to_device(&output_layer);
    //开始训练和测试
    testing(test_sample);
    cout << "run over 1" << endl;
    //清理内存
    //free_memory(test_sample,train_sample);
    release_layer(&input_layer);
    release_layer(&c1_convolution_layer);
    release_layer(&s2_pooling_layer);
    release_layer(&c3_convolution_layer);
    release_layer(&s4_pooling_layer);
    release_layer(&c5_convolution_layer);
    release_layer(&output_layer);
    // 释放device内存
    free_layer_device_memory(&input_layer);
    free_layer_device_memory(&c1_convolution_layer);
    free_layer_device_memory(&s2_pooling_layer);
    free_layer_device_memory(&c3_convolution_layer);
    free_layer_device_memory(&s4_pooling_layer);
    free_layer_device_memory(&c5_convolution_layer);
    free_layer_device_memory(&output_layer);
    
    return 0;
}
