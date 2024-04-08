#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float float32; // Define float32 as an alias for float
#include "hifi_model_trained_hdr.h" // Now include the header

float ***convolution(float ***image, int image_height, int image_width, int image_channels,
                     float ****filters, int num_filters, int kernel_height, int kernel_width,
                     int *stride, char *padding_mode, int *output_height, int *output_width, int *dilation, int num_groups, float* bias)
{
    // Check for valid input parameters
    if (image == NULL || filters == NULL || stride == NULL || padding_mode == NULL || output_height == NULL || output_width == NULL || dilation == NULL)
    {
        printf("Error: Invalid input parameters.\n");
        return NULL;
    }
    else if (  stride[0] <= 0 || stride[1] <= 0)
    {
        printf("Error: Stride Value should be greater than 0");
        return NULL;
    }
    else if (dilation[0] <= 0 || dilation[1] <= 0)
    {
        printf("Error: Dilation Value should be greater than 0");
        return NULL;
    }
    else if ((stride[0] > 1 && dilation[0] > 1) || (stride[1] > 1 && dilation[1] > 1) || (stride[0] > 1 && dilation[1] > 1) || (stride[1] > 1 && dilation[0] > 1))
    {
        printf("Error: Both stride and dilation can not be greater than 1");
        return NULL;
    }

    // Calculate padded dimensions
    int padded_height, padded_width;
    if (strcmp(padding_mode, "same") == 0)
    {
        padded_height = image_height + 2;
        padded_width = image_width + 2;
    }
    else if (strcmp(padding_mode, "valid") == 0)
    {
        padded_height = image_height;
        padded_width = image_width;
    }
    else
    {
        printf("Invalid padding mode. Please use 'same' or 'valid'.\n");
        return NULL;
    }
    if (padded_height <= 0 || padded_width <= 0)
    {
        printf("Error: Calculated output dimensions are invalid.\n");
        return NULL;
    }

    // Allocate memory for padded image
    float ***padded_image = (float ***)calloc(sizeof(float **), padded_height);
    if (padded_image == NULL)
    {
        printf("Error: Memory allocation failed for padded image.\n");
        return NULL;
    }
    for (int ph = 0; ph < padded_height; ph++)
    {
        padded_image[ph] = (float **)calloc(sizeof(float *), padded_width);
        if (padded_image[ph] == NULL)
        {
            printf("Error: Memory allocation failed for padded image.\n");
            return NULL;
        }
        for (int pw = 0; pw < padded_width; pw++)
        {
            padded_image[ph][pw] = (float *)calloc(sizeof(float), image_channels);
            if (padded_image[ph][pw] == NULL)
            {
                printf("Error: Memory allocation failed for padded image.\n");
                return NULL;
            }
        }
    }

    // Add padding to the input image
    for (int c = 0; c < image_channels; c++)
    {
        for (int i = 0; i < padded_height; i++)
        {
            for (int j = 0; j < padded_width; j++)
            {
                if (strcmp(padding_mode, "same") == 0)
                {
                    int row_offset = (kernel_height - 1) / 2;
                    int col_offset = (kernel_width - 1) / 2;
                    int original_row = i - row_offset;
                    int original_col = j - col_offset;
                    if (original_row < 0 || original_row >= image_height || original_col < 0 || original_col >= image_width)
                    {
                        // padded_image[c][i][j] = 0.0; // Pad with zeros
                        padded_image[i][j][c] = 0.0;
                    }
                    else
                    {
                        padded_image[i][j][c] = image[original_row][original_col][c];
                    }
                }
                else if (strcmp(padding_mode, "valid") == 0)
                {
                    padded_image[i][j][c] = image[i][j][c];
                }
            }
        }
    }

    int new_kernel_height = (kernel_height - 1) * dilation[0] + 1;
    int new_kernel_width = (kernel_width - 1) * dilation[1] + 1;
    if (new_kernel_height > padded_height || new_kernel_width > padded_width)
    {
        printf("Error: Filter size exceeds image size.\n");
        return NULL;
    }
    else if (new_kernel_height <= 0 || new_kernel_width <= 0)
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
    for (int i = 0; i < kernel_height; i++)
    {
        for (int j = 0; j < kernel_width; j++)
        {
            for (int c = 0; c < image_channels; c++)
            {
                for (int f = 0; f < num_filters; f++)
                {
                    int dilated_i = i * dilation[0];
                    int dilated_j = j * dilation[1];
                    if (dilated_i < new_kernel_height && dilated_j < new_kernel_width)
                    {
                        // dilated_filters[f][c][dilated_i][dilated_j] = filters[f][c][i][j];
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

    // Calculate output dimensions
    *output_height = ((padded_height - kernel_height - ((kernel_height - 1) * (dilation[0] - 1))) / stride[0]) + 1;
    *output_width = ((padded_width - kernel_width - ((kernel_width - 1) * (dilation[1] - 1))) / stride[1]) + 1;

    if (*output_height <= 0 || *output_width <= 0)
    {
        printf("Error: Calculated output dimensions are invalid.\n");
        return NULL;
    }


    float ***result = (float ***)calloc(sizeof(float **), *output_height);
    if (result == NULL)
    {
        printf("Error: Memory allocation failed for result array.\n");
        return NULL;
    }
    for (int h = 0; h < *output_height; h++)
    {
        result[h] = (float **)calloc(sizeof(float *), *output_height);
        if (result[h] == NULL)
        {
            printf("Error: Memory allocation failed for result array.\n");
            return NULL;
        }
        for (int w = 0; w < *output_width; w++)
        {
            result[h][w] = (float *)calloc(sizeof(float), num_filters);
            if (result[h][w] == NULL)
            {
                printf("Error: Memory allocation failed for result array.\n");
                return NULL;
            }
        }
    }

    if (num_filters % num_groups != 0)
    {
        printf("Error: Number of groups does not evenly divide the number of filters.\n");
        return NULL;
    }
    else if (num_groups <= 0 || num_groups > num_filters)
    {
        printf("Error: Number of groups must be a positive integer and not exceed the number of filters.\n");
        return NULL;
    }

    int group_num_filter = (num_filters / num_groups);

    // Perform convolution
    for (int group = 0; group < num_groups; group++)
    {
        int filter_start = group * group_num_filter;
        int filter_end = (group + 1) * group_num_filter;
        for (int f = filter_start; f < filter_end; f++)
        {
            for (int i = 0; i < *output_height; i++)
            {
                for (int j = 0; j < *output_width; j++)
                {
                    float sum = 0.0;
                    int group_kernel_channels = (image_channels / num_groups);
                    for (int c = 0; c < group_kernel_channels; c++)
                    {
                        for (int ki = 0; ki < new_kernel_height; ki++)
                        {
                            for (int kj = 0; kj < new_kernel_width; kj++)
                            {
                                // Adjust indices to consider dilation
                                int padded_i = i * stride[0] + ki;
                                int padded_j = j * stride[1] + kj;

                                if (padded_i < padded_height && padded_j < padded_width)
                                {
                                    sum += padded_image[padded_i][padded_j][c] * dilated_filters[ki][kj][c][f];
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

    // Free memory allocated for padded_image
    for (int ph = 0; ph < padded_height; ph++)
    {
        for (int pw = 0; pw < padded_width; pw++)
        {
            free(padded_image[ph][pw]);
        }
        free(padded_image[ph]);
    }
    free(padded_image);

    // Free memory for dilated filters
    for (int h = 0; h < new_kernel_height; h++)
    {
        for (int w = 0; w < new_kernel_width; w++)
        {
            for (int c = 0; c < image_channels; c++)
            {
                free(dilated_filters[h][w][c]);
            }
            free(dilated_filters[h][w]);
        }
        free(dilated_filters[h]);
    }
    free(dilated_filters);

    return result;
}

int main()
{
    // Define image
    int image_height = 5;
    int image_width = 5;
    int image_channels = 4;

    if (image_height <= 0 || image_width <= 0)
    {
        printf("Error: invalid image size.\n");
        return -1;
    }

    float ***image = (float ***)malloc(image_height * sizeof(float **));
    for (int h = 0; h < image_height; h++)
    {
        image[h] = (float **)malloc(image_width * sizeof(float *));
        for (int w = 0; w < image_width; w++)
        {
            image[h][w] = (float *)malloc(image_channels * sizeof(float));
        }
    }

    float image_values[5][5][4] =
       {
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
     {0.32961327, 0.91331383, 0.9779702, 0.79968478}}
};
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++)
        {
            for (int c = 0; c < image_channels; c++)
            {
                image[i][j][c] = image_values[i][j][c];
            }
        }
    }


    // Define stride and dilation
    int stride[2] = {1, 1};
    int dilation[2] = {1, 1};
    int num_groups = 1;
    // Perform convolution with "valid" padding
    int num_filters =  sizeof(conv2d_1_kernel[0][0][0]) / sizeof(conv2d_1_kernel[0][0][0][0]);
    int kernel_height = sizeof(conv2d_1_kernel)/sizeof(conv2d_1_kernel[0]);
    int kernel_width = sizeof(conv2d_1_kernel[0]) / sizeof(conv2d_1_kernel[0][0]);
    int kernel_channel = sizeof(conv2d_1_kernel[0][0]) / sizeof(conv2d_1_kernel[0][0][0]);
    //printf("%d",num_filters);
    float ****filters = (float ****)malloc(kernel_height * sizeof(float ***));
    for (int f = 0; f < kernel_height; f++)
    {
        filters[f] = (float ***)malloc(kernel_width * sizeof(float **));
        for (int c = 0; c < kernel_width; c++)
        {
            filters[f][c] = (float **)malloc(kernel_channel * sizeof(float *));
            for (int i = 0; i < kernel_channel; i++)
            {
                filters[f][c][i] = (float *)malloc(num_filters * sizeof(float));
            }
        }
    }
    
    for (int f = 0; f < kernel_height; f++)
    {
        for (int c = 0; c < kernel_width; c++)
        {
            for (int i = 0; i < kernel_channel; i++)
            {
                for (int j = 0; j < num_filters; j++)
                {
                    filters[f][c][i][j] = conv2d_1_kernel[f][c][i][j];
                }
            }
        }
    }
   float * bias = (float*)calloc(sizeof(float),num_filters);
   for(int i = 0;i<num_filters;i++)
   {
    bias[i] = conv2d_1_bias[i];
   }
   for(int i = 0;i<num_filters;i++)
   {
    printf("%.6f ",bias[i]);
   }


   // Perform convolution with "valid" padding 
    int output_height;
    int output_width;
    float ***result = convolution(image, image_height, image_width, image_channels,
                                  filters, num_filters, kernel_height, kernel_width,
                                  stride, "valid", &output_height, &output_width, dilation, num_groups, bias);

    if (result != NULL)
    {
        // Print the result
        printf("Output Height: %d, Output Width: %d\n", output_height, output_width);
        
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                for (int f = 0; f < num_filters; f++)
                {
                    printf("%.8f ", result[i][j][f]);
                }
                printf("\n");
            }
            printf("\n");
        }
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
    for (int i = 0; i < kernel_height; i++)
    {
        for (int j = 0; j < kernel_width; j++)
        {
            for (int c = 0; c < image_channels; c++)
            {
                free(filters[i][j][c]);
            }
            free(filters[i][j]);
        }
        free(filters[i]);
    }
    free(filters);

    return 0;
}