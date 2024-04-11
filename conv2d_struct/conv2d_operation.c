#include <math.h>
#include <stdio.h>
#include<stdlib.h>
#include "conv2d_config.h"

float ****kernel_dilation(float ****filters, int *kernel_height, int *kernel_width, int image_channels, int num_filters, int *dilation)
{
    int new_kernel_height = (*kernel_height - 1) * dilation[0] + 1;
    int new_kernel_width = (*kernel_width - 1) * dilation[1] + 1;
    if (new_kernel_height <= 0 || new_kernel_width <= 0)
    {
        printf("Error: Calculated kernel dimensions are invalid.\n");
        return NULL;
    }

    // Allocate memory for dilated filters
    float ****dilated_filters = (float ****)calloc(sizeof(float ***), new_kernel_height);
    if (dilated_filters == NULL)
    {
        printf("Error: Memory allocation failed for dilated filter.\n");
        return NULL;
    }
    for (int h = 0; h < new_kernel_height; h++)
    {
        dilated_filters[h] = (float ***)calloc(sizeof(float **), new_kernel_width);
        if (dilated_filters[h] == NULL)
        {
            printf("Error: Memory allocation failed for dilated filter.\n");
            return NULL;
        }

        for (int w = 0; w < new_kernel_width; w++)
        {
            dilated_filters[h][w] = (float **)calloc(sizeof(float *), image_channels);
            if (dilated_filters[h][w] == NULL)
            {
                printf("Error: Memory allocation failed for dilated filter.\n");
                return NULL;
            }
            for (int c = 0; c < image_channels; c++)
            {
                dilated_filters[h][w][c] = (float *)calloc(sizeof(float), num_filters);
                if (dilated_filters[h][w][c] == NULL)
                {
                    printf("Error: Memory allocation failed for dilated filter.\n");
                    return NULL;
                }
            }
        }
    }

    // Populate the dilated filters with values from the original filters
    for (int i = 0; i < *kernel_height; i++)
    {
        for (int j = 0; j < *kernel_width; j++)
        {
            for (int c = 0; c < image_channels; c++)
            {
                for (int f = 0; f < num_filters; f++)
                {
                    int dilated_i = i * dilation[0];
                    int dilated_j = j * dilation[1];
                    if (dilated_i < new_kernel_height && dilated_j < new_kernel_width)
                    {
                        dilated_filters[dilated_i][dilated_j][c][f] = filters[i][j][c][f];
                    }
                    else
                    {
                        printf("Error: Dilated indices out of bounds.\n");
                        return NULL;
                    }
                }
            }
        }
    }

    *kernel_height = new_kernel_height;
    *kernel_width = new_kernel_width;

    printf("new_kernel_height %d \n",new_kernel_height);
    printf("new_kernel_width %d \n",new_kernel_width);

    return dilated_filters;
}
float*** conv2d_execution(float ***image, int image_height, int image_width, int image_channels,
                     float ****filters, int num_filters, int kernel_height, int kernel_width, 
                     int output_height, int output_width, float* bias,
                     int num_groups, int *stride,int* dilation)
{
   
    kernel_dilation(filters,&kernel_height,&kernel_width,image_channels,num_filters,dilation);
    float ***result = (float ***)calloc(sizeof(float **), output_height);
    for (int h = 0; h < output_height; h++)
    {
        result[h] = (float **)calloc(sizeof(float *),output_height);
        for (int w = 0; w < output_width; w++)
        {
            result[h][w] = (float *)calloc(sizeof(float), num_filters);
           
        }
    }
int group_num_filter = (num_filters / num_groups);

    // Perform convolution
    for (int group = 0; group < num_groups; group++)
    {
        int filter_start = group * group_num_filter;
        int filter_end = (group + 1) * group_num_filter;
        for (int f = filter_start; f < filter_end; f++)
        {
            for (int i = 0; i < output_height; i++)
            {
                for (int j = 0; j < output_width; j++)
                {
                    float sum = 0.0;
                    int group_kernel_channels = (image_channels / num_groups);
                    for (int c = 0; c < group_kernel_channels; c++)
                    {
                        for (int ki = 0; ki < kernel_height; ki++)
                        {
                            for (int kj = 0; kj < kernel_width; kj++)
                            {
                                // Adjust indices to consider dilation
                                int padded_i = i * stride[0] + ki;
                                int padded_j = j * stride[1] + kj;

                                if (padded_i < image_height && padded_j < image_width)
                                {
                                    sum += image[padded_i][padded_j][c] * filters[ki][kj][c][f];
                                }
                            }
                        }
                    }
                    sum += bias[f];
                    result[i][j][f] = sum;
                }
            }
       }
    }
   
    return result;
}
                     
                     