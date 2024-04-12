#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "transpconv2d_config.h"

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
            dilated_filters[h][w] = (float **)calloc(sizeof(float *), num_filters);
            if (dilated_filters[h][w] == NULL)
            {
                printf("Error: Memory allocation failed for dilated filter.\n");
                return NULL;
            }
            for (int f = 0; f < num_filters; f++)
            {
                dilated_filters[h][w][f] = (float *)calloc(sizeof(float), image_channels);
                if (dilated_filters[h][w][f] == NULL)
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
            for (int f = 0; f < num_filters; f++)
            {
                for (int c = 0; c < image_channels; c++)
                {
                    int dilated_i = i * dilation[0];
                    int dilated_j = j * dilation[1];
                    if (dilated_i < new_kernel_height && dilated_j < new_kernel_width)
                    {
                        dilated_filters[dilated_i][dilated_j][f][c] = filters[i][j][f][c];
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

    return dilated_filters;
}

float ***transpconv2d_execution(float ***image, int image_height, int image_width, int image_channels,
                                float ****filters, int num_filters, int kernel_height, int kernel_width,
                                int *output_height, int *output_width, float *bias,
                                int *stride, int *dilation)
{

    if (image == NULL || filters == NULL || stride == NULL || dilation == NULL)
    {
        printf("Error: Invalid input parameters.\n");
        return NULL;
    }
    else if (stride[0] <= 0 || stride[1] <= 0)
    {
        printf("Error: Stride Value should be greater than 0.\n");
        return NULL;
    }
    else if (dilation[0] <= 0 || dilation[1] <= 0)
    {
        printf("Error: Dilation Value should be greater than 0.\n");
        return NULL;
    }
    else if (dilation[0] != dilation[1])
    {
        printf("Error: Dilation Values should be equal to each than.\n");
        return NULL;
    }
    else if ((stride[0] > 1 && dilation[0] > 1) || (stride[1] > 1 && dilation[1] > 1) || (stride[0] > 1 && dilation[1] > 1) || (stride[1] > 1 && dilation[0] > 1))
    {
        printf("Error: Both stride and dilation can not be greater than 1.\n");
        return NULL;
    }
    else if (kernel_height <= 0 || kernel_width <= 0)
    {
        printf("Error: Given kernel dimensions are invalid.\n");
        return NULL;
    }

    if (kernel_height > 1 && kernel_width > 1)
    {
        filters = kernel_dilation(filters, &kernel_height, &kernel_width, image_channels, num_filters, dilation);
    }

    if (stride[0] > kernel_height)
    {
        *output_height = (image_height * stride[0]);
    }
    else
    {
        *output_height = ((image_height - 1) * stride[0]) + (kernel_height);
    }
    if (stride[1] > kernel_width)
    {
        *output_width = (image_width * stride[1]);
    }
    else
    {
        *output_width = ((image_width - 1) * stride[1]) + (kernel_width);
    }

    if (*output_height <= 0 || *output_width <= 0)
    {
        printf("Error: Calculated output dimensions are invalid.\n");
        return NULL;
    }

    float ***result = (float ***)calloc(sizeof(float **), *output_height);
    for (int h = 0; h < *output_height; h++)
    {
        result[h] = (float **)calloc(sizeof(float *), *output_width);
        for (int w = 0; w < *output_width; w++)
        {
            result[h][w] = (float *)calloc(sizeof(float), num_filters);
        }
    }

    int res_width = 0;
    int res_height = 0;


    // Perform transpose convolution
    for (int f = 0; f < num_filters; f++)
    {
        for (int i = 0; i < image_height; i++)
        {
            for (int j = 0; j < image_width; j++)
            {
                float temp = 0.0;
                for (int c = 0; c < image_channels; c++)
                {
                    for (int ki = 0; ki < kernel_height; ki++)
                    {
                        for (int kj = 0; kj < kernel_width; kj++)
                        {

                            temp = image[i][j][c] * filters[ki][kj][f][c];
                            res_height = (stride[0] * i) + ki;
                            res_width = (stride[1] * j) + kj;
                            result[res_height][res_width][f] += temp;
                        }
                    }
                }
            }
        }
    }

    // Add bias
    for (int h = 0; h < *output_height; h++)
    {
        for (int w = 0; w < *output_width; w++)
        {
            for (int f = 0; f < num_filters; f++)
            {
                result[h][w][f] += bias[f];
            }
        }
    }

    return result;
}