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
    config->num_groups = 2;

    int image_height = 5;
    int image_width = 5;
    int image_channel = 4;

    float image_values[5][5][4] =
        {{{-0.868075, 0.6323161, -1.029794, -1.1226166},
          {0.6819557, 0.680788, -1.353582, 0.235164},
          {1.0329384, -1.8830103, -0.00931417, 0.384478},
          {0.05629285, 1.3480552, 0.38766965, -0.15001388},
          {-0.16527863, -0.8368426, -0.59166497, -1.1044368}},

         {{-0.9113604, -0.7691372, 0.3107127, -0.05129186},
          {-0.24027742, -2.1465328, -0.10581684, 0.06907097},
          {-0.01357317, 0.36070526, 0.74174374, 0.14447758},
          {0.695339, -0.14691441, 1.8235788, -1.8128419},
          {-0.9706362, 0.83843094, 1.2090212, -1.1345373}},

         {{-0.79966944, -1.2127473, -3.0084255, 0.4083387},
          {0.5612697, -0.96531403, -0.47506377, 0.81423956},
          {-0.30977178, 1.166958, 1.0125041, -0.15710047},
          {-1.5220784, 0.7308277, 0.85494643, -1.1234224},
          {0.3746431, 1.6002654, 0.1273398, -1.5829618}},

         {{-0.56852317, -0.48928127, -0.70081204, 0.07127578},
          {0.12041917, -0.5626022, -1.5719923, 0.7154496},
          {0.36430353, 0.16553247, -2.50368, -0.03518617},
          {0.37823078, -0.5136497, 1.3139606, 0.30274013},
          {-0.41425133, 2.1505494, 0.6919361, -1.4027561}},

         {{1.65403, 0.9130174, 1.1149281, 0.31310338},
          {2.3869069, -1.0736752, 0.2522008, 0.19523281},
          {-0.12904295, -1.9827739, 0.7088773, 0.09952194},
          {-0.57468253, -1.0605109, -0.08598536, 0.83258885},
          {2.224367, -1.7306535, -0.69799155, 1.267176}}};

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

    config->num_filter = sizeof(filter[0][0][0]) / sizeof(filter[0][0][0][0]);
    config->kernel_height = sizeof(filter) / sizeof(filter[0]);
    config->kernel_width = sizeof(filter[0]) / sizeof(filter[0][0]);
    config->image_channels = sizeof(filter[0][0]) / sizeof(filter[0][0][0]);

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

    config->bias = malloc(config->num_filter * sizeof(float));
    for (int i = 0; i < config->num_filter; i++)
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
                config->filter[i][j][c] = malloc(config->num_filter * sizeof(float));
                for (int n = 0; n < config->num_filter; n++)
                {
                    config->filter[i][j][c][n] = filter[i][j][c][n];
                }
            }
        }
    }

    int output_height;
    int output_width;

    // Your code to calculate and fill the output tensor goes here

    float ***output = conv2d_execution(image, image_height, image_width, image_channel,
                                       config->filter, config->num_filter, config->kernel_height, config->kernel_width, &output_height, &output_width, config->bias,
                                       config->num_groups, config->stride, config->dilation);
    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            for (int f = 0; f < config->num_filter; f++)
            {
                printf("%.9lf ", output[i][j][f]);
            }
            printf("\n");
        }
        printf("\n");
    }
    // Free memory allocated for image
    for (int h = 0; h < image_height; h++)
    {
        for (int w = 0; w < image_width; w++)
        {
            free(image[h][w]);
        }
        free(image[h]);
    }
    free(image);

    // Free memory allocated for filters
    for (int i = 0; i < config->kernel_height; i++)
    {
        for (int j = 0; j < config->kernel_width; j++)
        {
            for (int c = 0; c < image_channel; c++)
            {
                free(config->filter[i][j][c]);
            }
            free(config->filter[i][j]);
        }
        free(config->filter[i]);
    }
    free(config->filter);

    return 0;
}
