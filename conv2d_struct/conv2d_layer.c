#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "conv2d_config.h"
typedef float float32;
#include "hifi_model_trained_hdr.h"

int main()
{
    int dilation_values[2] = {1, 1};
    int stride_values[2] = {1, 1};
    conv2d_config *config = malloc(sizeof(conv2d_config));
    config->num_filters = sizeof(filter[0][0][0]) / sizeof(filter[0][0][0][0]);
    config->kernel_height = sizeof(filter) / sizeof(filter[0]);
    config->kernel_width = sizeof(filter[0]) / sizeof(filter[0][0]);
    config->image_channels = sizeof(filter[0][0]) / sizeof(filter[0][0][0]);
    config->num_groups = 1;
    config->stride = malloc(2 * sizeof(int));
    for (int i = 0; i < 2; i++)
    {
        config->stride[i] = stride_values[i];
    }
    config->dilation = malloc(2 * sizeof(int));
    for (int i = 0; i < 2; i++)
    {
        config->dilation[i] = dilation_values[i];
    }

    config->bias = malloc(config->num_filters * sizeof(float));
    for (int i = 0; i < config->num_filters; i++)
    {
        config->bias[i] = conv2d_bias[i];
    }
    config->filter = malloc(config->kernel_height * sizeof(float ***));
    for (int i = 0; i < config->kernel_height; i++)
    {
        config->filter[i] = malloc(config->kernel_width * sizeof(float **));
        for (int j = 0; j < config->kernel_width; j++)
        {
            config->filter[i][j] = malloc(config->image_channels * sizeof(float *));
            for (int c = 0; c < config->image_channels; c++)
            {
                config->filter[i][j][c] = malloc(config->num_filters * sizeof(float));
                for (int n = 0; n < config->num_filters; n++)
                {
                    config->filter[i][j][c][n] = filter[i][j][c][n];
                }
            }
        }
    }

    int image_height = 5;
    int image_width = 5;
    int image_channel = 4;

    float image_values[5][5][4] = {
        {{0.08735558, 0.02516036, 0.46663856, 0.97814474},
         {0.03029301, 0.09175156, 0.93132864, 0.46998856},
         {0.99934153, 0.94569276, 0.71094907, 0.8327293},
         {0.35891874, 0.64470921, 0.58847226, 0.49062343},
         {0.85021439, 0.31821496, 0.36390768, 0.80509833}},

        {{0.22988014, 0.20926591, 0.42101532, 0.68564857},
         {0.96302982, 0.80052675, 0.31212618, 0.81744366},
         {0.65513446, 0.36521154, 0.55334657, 0.61804096},
         {0.53257528, 0.30959641, 0.31400056, 0.6382605},
         {0.95577655, 0.86354521, 0.76594888, 0.73937023}},

        {{0.83042559, 0.44858606, 0.36167685, 0.55149401},
         {0.52312661, 0.4793265, 0.7249799, 0.14329753},
         {0.42841868, 0.90731582, 0.3450547, 0.31774168},
         {0.43397553, 0.37322366, 0.06931127, 0.12144709},
         {0.94163643, 0.85786455, 0.2100153, 0.77737175}},

        {{0.40395135, 0.85768075, 0.45353044, 0.76452005},
         {0.11921593, 0.25197029, 0.10307968, 0.39682513},
         {0.2988042, 0.9675748, 0.83398327, 0.51014497},
         {0.69075402, 0.35799987, 0.01392934, 0.55891134},
         {0.84453829, 0.92202443, 0.23126256, 0.15384924}},

        {{0.61583206, 0.96013963, 0.50585135, 0.35911717},
         {0.8400894, 0.06741817, 0.57265725, 0.42650627},
         {0.60680911, 0.45548849, 0.14715885, 0.68030919},
         {0.79142864, 0.94768057, 0.2217362, 0.82952721},
         {0.32961327, 0.91331383, 0.9779702, 0.79968478}}};

    float ***image = (float ***)malloc(image_height * sizeof(float **));
    for (int h = 0; h < image_height; h++)
    {
        image[h] = (float **)malloc(image_width * sizeof(float *));
        for (int w = 0; w < image_width; w++)
        {
            image[h][w] = (float *)malloc(image_channel * sizeof(float));
        }
    }
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            for (int c = 0; c < image_channel; c++)
            {
                image[i][j][c] = image_values[i][j][c];
            }
        }
    }

    int output_height = ((image_height - config->kernel_height - ((config->kernel_height - 1) * (config->dilation[0] - 1))) / config->stride[0]) + 1;
    int output_width = ((image_width - config->kernel_width - ((config->kernel_width - 1) * (config->dilation[1] - 1))) / config->stride[1]) + 1;
    // Your code to calculate and fill the output tensor goes here

    float ***output = conv2d_execution(image, image_height, image_width, image_channel,
                                       config->filter, config->num_filters, config->kernel_height, config->kernel_width, output_height, output_width, config->bias,
                                       config->num_groups, config->stride, config->dilation);
    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            for (int f = 0; f < config->num_filters; f++)
            {
                printf("%lf ", output[i][j][f]);
            }
            printf("\n");
        }
        printf("\n");
    }
    return 0;
}
