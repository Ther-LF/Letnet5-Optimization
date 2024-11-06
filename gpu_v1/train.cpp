#include "util.h"
#include "structure.h"
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
void convolution_calcu(double* input_map_data, int input_map_width, int input_map_height, double* kernel_data, int kernel_width, int kernel_height, double* result_map_data, int result_map_width, int result_map_height)
{
    log_exec("convolution_calcu");

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
            convolution_calcu(
                pre_layer->map[j].data, pre_layer->map_width, pre_layer->map_height,
                cur_layer->kernel[index_layer].weight, cur_layer->kernel_width, cur_layer->kernel_height,
                cur_layer->map_common, cur_layer->map_width, cur_layer->map_height
            );
        }
        for (int k = 0; k < layer_size; k++)
        {
            cur_layer->map[i].data[k] = activation_function::tan_h(cur_layer->map_common[k] + cur_layer->map[i].bias);
        }
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
/*
    全连接层反向更新
*/
void fully_connection_backward_propagation(Layer* cur_layer, Layer* pre_layer)
{   log_exec("fully_connection_backward_propagation");
    for (int i = 0; i < pre_layer->map_count; i++)
    {
        pre_layer->map[i].error[0] = 0.0;
        for (int j = 0; j < cur_layer->map_count; j++)
        {
            pre_layer->map[i].error[0] += cur_layer->map[j].error[0] * cur_layer->kernel[i * cur_layer->map_count + j].weight[0];
        }
        pre_layer->map[i].error[0] *= activation_function::d_tan_h(pre_layer->map[i].data[0]);
    }
    //更新c5 --> output kernel delta_weight
    for (int i = 0; i < pre_layer->map_count; i++)
    {
        for (int j = 0; j < cur_layer->map_count; j++)
        {
            cur_layer->kernel[i * cur_layer->map_count + j].delta_weight[0] += cur_layer->map[j].error[0] * pre_layer->map[i].data[0];
        }
    }
    //更新output delta_bias
    for (int i = 0; i < cur_layer->map_count; i++)
    {
        cur_layer->map[i].delta_bias += cur_layer->map[i].error[0];
    }
}
void convolution_backward_propagation(Layer* cur_layer, Layer* pre_layer, bool* connection_table) //checked checked 2.0 fixed
{
    log_exec("convolution_backward_propagation");

    int connected_index = 0;
    int pre_layer_mapsize = pre_layer->map_height * pre_layer->map_width;
    //更新S4 error
    for (int i = 0; i < pre_layer->map_count; i++)
    {
        memset(pre_layer->map_common, 0, sizeof(double) * pre_layer_mapsize);
        for (int j = 0; j < cur_layer->map_count; j++)
        {
            connected_index = i * cur_layer->map_count + j;
            if (connection_table != NULL && !connection_table[connected_index])
            {
                continue;
            }
            for (int n = 0; n < cur_layer->map_height; n++) //fixed cur_layer->kernel_height -->>  cur_layer->map_height
            {
                for (int m = 0; m < cur_layer->map_width; m++)
                {
                    int valued_index = n * cur_layer->map_width + m;
                    double error = cur_layer->map[j].error[valued_index];
                    for (int kernel_y = 0; kernel_y < cur_layer->kernel_height; kernel_y++)
                    {
                        for (int kernel_x = 0; kernel_x < cur_layer->kernel_width; kernel_x++)
                        {
                            int index_convoltion_map = (n + kernel_y) * pre_layer->map_width + m + kernel_x;
                            int index_kernel = connected_index;
                            int index_kernel_weight = kernel_y * cur_layer->kernel_width + kernel_x;
                            pre_layer->map_common[index_convoltion_map] += error * cur_layer->kernel[index_kernel].weight[index_kernel_weight];
                        }
                    }
                }
            }
        }
        for (int k = 0; k < pre_layer_mapsize; k++)
        {
            pre_layer->map[i].error[k] = pre_layer->map_common[k] * activation_function::d_tan_h(pre_layer->map[i].data[k]);
            //pre_layer->map[i].error[k] = pre_layer->map_common[k] * activation_func::dtan_h(prev_layer->map[i].data[k]); source
        }
    }
    //更新 S_x ->> C_x kernel 的 delta_weight
    for (int i = 0; i < pre_layer->map_count; i++)
    {
        for (int j = 0; j < cur_layer->map_count; j++)
        {
            connected_index = i * cur_layer->map_count + j;
            if (connection_table != NULL && !connection_table[connected_index])
            {
                continue;
            }
            //fixed cur_layer->map[i] -->> cur_layer->map[j]
            convolution_calcu(
                pre_layer->map[i].data, pre_layer->map_width, pre_layer->map_height,
                cur_layer->map[j].error, cur_layer->map_width, cur_layer->map_height,
                cur_layer->kernel[connected_index].delta_weight, cur_layer->kernel_width, cur_layer->kernel_height
            );
        }
    }
    //更新C_x 的delta_bias
    int cur_layer_mapsize = cur_layer->map_height * cur_layer->map_width;
    for (int i = 0; i < cur_layer->map_count; i++)
    {
        double delta_sum = 0.0;
        for (int j = 0; j < cur_layer_mapsize; j++)
        {
            delta_sum += cur_layer->map[i].error[j];
        }
        cur_layer->map[i].delta_bias += delta_sum;
    }
}
void max_pooling_backward_propagation(Layer* cur_layer, Layer* pre_layer)  //checked
{
    log_exec("max_pooling_backward_propagation");

    int cur_layer_mapwidth = cur_layer->map_width;
    int cur_layer_mapheight = cur_layer->map_height;
    int pre_layer_mapwidth = pre_layer->map_width;
    for (int k = 0; k < cur_layer->map_count; k++)
    {
        for (int i = 0; i < cur_layer_mapheight; i++)
        {
            for (int j = 0; j < cur_layer_mapwidth; j++)
            {
                int index_row = 2 * i;
                int index_col = 2 * j;
                double max_value = pre_layer->map[k].data[index_row * pre_layer_mapwidth + index_col];
                for (int n = 2 * i; n < 2 * (i + 1); n++)
                {
                    for (int m = 2 * j; m < 2 * (j + 1); m++)
                    {
                        if (pre_layer->map[k].data[n * pre_layer_mapwidth + m] > max_value)
                        {
                            index_row = n;
                            index_col = m;
                            max_value = pre_layer->map[k].data[n * pre_layer_mapwidth + m];
                        }
                        else
                        {
                            pre_layer->map[k].error[n * pre_layer_mapwidth + m] = 0.0;
                        }
                    }
                }
                pre_layer->map[k].error[index_row * pre_layer_mapwidth + index_col] = cur_layer->map[k].error[i * cur_layer_mapwidth + j] * activation_function::d_tan_h(max_value);
            }
        }
    }
}
void backward_propagation(double* label)  //checked
{
    log_exec("backward_propagation");

    //前面并未初始化输出层error，在此初始化
    for (int i = 0; i < output_layer.map_count; i++)
    {
        output_layer.map[i].error[0] = loss_function::d_cal_loss(output_layer.map[i].data[0], label[i]) * activation_function::d_tan_h(output_layer.map[i].data[0]);
    }
    fully_connection_backward_propagation(&output_layer, &c5_convolution_layer);
    convolution_backward_propagation(&c5_convolution_layer, &s4_pooling_layer, NULL);
    max_pooling_backward_propagation(&s4_pooling_layer, &c3_convolution_layer);
    convolution_backward_propagation(&c3_convolution_layer, &s2_pooling_layer, connection_table);
    max_pooling_backward_propagation(&s2_pooling_layer, &c1_convolution_layer);
    convolution_backward_propagation(&c1_convolution_layer, &input_layer, NULL);
}
//梯度下降
inline double gradient_descent(double para, double delta_para, double learning_rate) //checked
{
    log_exec("init_kernel");
    //     cout<<"exec fun -->> [gradient_descent]"<<endl;
    return para - learning_rate * delta_para;
}
/*
    梯度下降法
    更新某layer的 kernel 的weight 和 map的 bias
*/
void update_param(Layer* layer, double learning_rate) //checked and fixed
{
    log_exec("update_param");

    int kernel_size = layer->kernel_height * layer->kernel_width;
    //update weight
    for (int i = 0; i < layer->kernel_count; i++)
    {
        for (int k = 0; k < kernel_size; k++)
        {
            layer->kernel[i].weight[k] = gradient_descent(layer->kernel[i].weight[k], layer->kernel[i].delta_weight[k] / batch_size, learning_rate);
        }
    }
    for (int i = 0; i < layer->map_count; i++)  //fixed add ：/ batch_size
    {
        layer->map[i].bias = gradient_descent(layer->map[i].bias, layer->map[i].delta_bias / batch_size, learning_rate);
    }
}

void update_all_layer_param(double learning_rate)
{
    log_exec("update_all_layer_param");

    update_param(&c1_convolution_layer, learning_rate);
    update_param(&s2_pooling_layer, learning_rate);
    update_param(&c3_convolution_layer, learning_rate);
    update_param(&s4_pooling_layer, learning_rate);
    update_param(&c5_convolution_layer, learning_rate);
    update_param(&output_layer, learning_rate);
}
void training(Sample* samples, double learning_rate)
{
    log_exec("training");

    int batch_count = samples->sample_count / batch_size;
    int single_sample_size = samples->sample_height * samples->sample_width * sizeof(double);
    cout << "training started!!!" << endl;
    cout << "sample count : " << samples->sample_count << "\t batch size :" << batch_size << " \t batch count : " << batch_count << endl;
    for (int i = 0; i < batch_count; i++)
    {

        reset_all_layer(); //重新初始化各层map的delta bias 和 kernel 的 delta weight
        for (int j = 0; j < batch_size; j++)
        {
            int sample_index = i * batch_size + j;
            memcpy(input_layer.map[0].data, samples[sample_index].data, single_sample_size);

            //show_map(input_layer.map[0],input_layer.map_height,input_layer.map_width); //test

            forward_propagation();
            backward_propagation(samples[sample_index].label);
        }
        update_all_layer_param(learning_rate);
        if (i % 1500 == 0)
        {
            // show_pic(samples[i]); //test

            cout << "training data process : " << i / (1.0 * batch_count) * 100.0 << " %" << endl;
        }
    }
    cout << "training data process : 100 %" << endl;
    cout << "training data over!!!!" << endl;
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
        memcpy(input_layer.map[0].data, samples[i].data, single_sample_size);
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
        //cout << "label value : " << result_label << "\t test value : " << result_test << endl;
        result_test_label_matrix[result_test * class_count + result_label] ++;
        if (i % 2500 == 0)
        {
            cout << "testing data process : " << i / (1.0 * samples->sample_count) * 100.0 << " %" << endl;
        }
    }

    //输出结果矩阵
    cout << "testing over !!! success rate : " << (1.0) * success_count / samples->sample_count << endl;
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

void release_layer(Layer* layer)
{
    log_exec("release_layer");

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

void free_memory(Sample *test_sample, Sample *train_sample)
{
    log_exec("free_memory");

    for (int i = 0; i < train_sample_count; i++) {
        free(train_sample[i].data);
        free(train_sample[i].label);
        train_sample[i].data = NULL;
        train_sample[i].label = NULL;
    }
    free(train_sample);

    for (int i = 0; i < test_sample_count; i++) {
        free(test_sample[i].data);
        free(test_sample[i].label);
        test_sample[i].data = NULL;
        test_sample[i].label = NULL;
    }
    free(test_sample);
}





int main()
{
    
    log_exec("MAIN");
    double learning_rate = 0.018;
    //初始化训练集样本
    auto* train_sample = static_cast<Sample *>(malloc(train_sample_count * sizeof(Sample)));
    memset(train_sample, 0, train_sample_count * sizeof(Sample));
    train_sample->sample_height = height;
    train_sample->sample_width = width;
    train_sample->sample_count = train_sample_count;
    const char* train_sample_path = "../data/train-images.idx3-ubyte";
    const char* train_label_path = "../data/train-labels.idx1-ubyte";
    load_mnist_data(train_sample, train_sample_path, 60000);
    load_mnist_label(train_sample, train_label_path, 60000);
    //初始化各层
    //init_layer(*layer, prelayer_map_count, map_count, kernel_width, kernel_height, map_width, map_height, is_pooling)
    init_layer(&input_layer, 0, 1, 0, 0, 32, 32, false);
    init_layer(&c1_convolution_layer, 1, 6, 5, 5, 28, 28, false);
    init_layer(&s2_pooling_layer, 1, 6, 1, 1, 14, 14, true);
    init_layer(&c3_convolution_layer, 6, 16, 5, 5, 10, 10, false);
    init_layer(&s4_pooling_layer, 1, 16, 1, 1, 5, 5, true);
    init_layer(&c5_convolution_layer, 16, 120, 5, 5, 1, 1, false);
    init_layer(&output_layer, 120, 10, 1, 1, 1, 1, false);
    //开始训练和测试
    training(train_sample, learning_rate);
    cout << "run over 1" << endl;
    // 保存权重
    save_layer(input_layer, "./weight/input_layer");
    save_layer(c1_convolution_layer, "./weight/c1_convolution_layer");
    save_layer(s2_pooling_layer, "./weight/s2_pooling_layer");
    save_layer(c3_convolution_layer, "./weight/c3_convolution_layer");
    save_layer(s4_pooling_layer, "./weight/s4_pooling_layer");
    save_layer(c5_convolution_layer, "./weight/c5_convolution_layer");
    save_layer(output_layer, "./weight/output_layer");
    //清理内存
    //free_memory(test_sample,train_sample);
    release_layer(&input_layer);
    release_layer(&c1_convolution_layer);
    release_layer(&s2_pooling_layer);
    release_layer(&c3_convolution_layer);
    release_layer(&s4_pooling_layer);
    release_layer(&c5_convolution_layer);
    release_layer(&output_layer);
    
    return 0;
}
