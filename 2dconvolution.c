#include <stdio.h>
#include <stdlib.h>
#include <string.h>

float ***convolution(float ***image, int image_height, int image_width, int image_channels,
                     float ****filters, int num_filters, int kernel_height, int kernel_width,
                     int *stride, char *padding_mode, int *output_height, int *output_width, int *dilation, int num_groups, float bias)
{
    // Check for valid input parameters
    if (image == NULL || filters == NULL || stride == NULL || padding_mode == NULL || output_height == NULL || output_width == NULL || dilation == NULL)
    {
        printf("Error: Invalid input parameters.\n");
        return NULL;
    }
    else if (stride[0] <= 0 || stride[1] <= 0)
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
                    sum += bias;
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
    int image_channels = 3;

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

    float image_values[5][5][3] =
        {{{1.1306248, -0.25314054, 3.290393},
          {-1.7246673, 1.3534538, -0.3121599},
          {-0.05556452, -0.20544375, -0.56706166},
          {-0.67606163, 0.3130979, -0.19556145},
          {-0.6345593, -0.43959042, -1.4159933}},

         {{-1.0060836, -0.26721567, -0.02783969},
          {1.1502476, -0.51878625, 0.72890097},
          {1.6515682, -0.9732839, -1.7348298},
          {0.95341325, -0.01147437, -1.638561},
          {0.1413229, -0.82040286, 0.7101485}},

         {{-0.2470776, 0.00438557, 0.09976939},
          {-0.38960364, 1.4298465, 0.9944432},
          {-2.4888053, -1.306025, 0.8938619},
          {-0.72093546, -0.50181943, -1.1170171},
          {-1.4915686, -0.08969079, -0.76380324}},

         {{0.3822652, -0.02399319, 1.2866206},
          {-1.6156209, 0.07037552, 0.04747601},
          {-0.16574988, 0.7411207, 0.45561045},
          {0.32828468, 0.42320758, -1.2261853},
          {-0.40239573, -0.11335092, -1.4435183}},

         {{1.0356123, -0.14511828, -0.47873953},
          {-0.24055727, 2.7634437, -0.9846009},
          {1.357394, 1.0843923, 1.048625},
          {-1.9879742, 0.7724398, -0.15974393},
          {-0.6112168, -1.0985042, -0.32617888}}};

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

    // Define filters
    int num_filters = 1; // Changed to match filter_values dimensions
    int kernel_channels = 3;
    int kernel_height = 3; // Changed to match filter_values dimensions
    int kernel_width = 3;  // Changed to match filter_values dimensions
    if (kernel_height <= 0 || kernel_width <= 0)
    {
        printf("Error: invalid kernel size.\n");
        return -1;
    }
    if (image_channels != kernel_channels)
    {
        printf("Error: Image and kernel must have the same number of channels.\n");
        return -1;
    }

    float ****filters = (float ****)malloc(kernel_height * sizeof(float ***));
    for (int f = 0; f < kernel_height; f++)
    {
        filters[f] = (float ***)malloc(kernel_channels * sizeof(float **));
        for (int c = 0; c < kernel_channels; c++)
        {
            filters[f][c] = (float **)malloc(kernel_height * sizeof(float *));
            for (int i = 0; i < kernel_height; i++)
            {
                filters[f][c][i] = (float *)malloc(kernel_width * sizeof(float));
            }
        }
    }

    float filter_values[3][3][3][1] =
        {
            {{{0.22829092}, {-0.00311419}, {0.3199622}},
             {{0.23403156}, {0.36613286}, {-0.33458662}},
             {{0.4009111}, {0.22241455}, {-0.10484788}}},
            {{{0.21512544}, {-0.08512771}, {0.28911948}},
             {{0.31168872}, {0.2567218}, {0.3718804}},
             {{0.31413615}, {0.28962827}, {0.09851223}}},
            {{{0.39324045}, {0.25920504}, {-0.19405262}},
             {{-0.0487029}, {-0.39086708}, {0.11018586}},
             {{-0.0494256}, {-0.3000655}, {-0.21036112}}}};

    for (int f = 0; f < kernel_height; f++)
    {
        for (int c = 0; c < kernel_channels; c++)
        {
            for (int i = 0; i < kernel_height; i++)
            {
                for (int j = 0; j < kernel_width; j++)
                {
                    filters[f][c][i][j] = filter_values[f][c][i][j];
                }
            }
        }
    }


    // Define stride and dilation
    int stride[2] = {1, 1};
    int dilation[2] = {1, 1};
    int num_groups = 1;
    float bias = 0.0;

    // Perform convolution with "valid" padding
    int output_height, output_width;
    float ***result = convolution(image, image_height, image_width, image_channels,
                                  filters, num_filters, kernel_height, kernel_width,
                                  stride, "valid", &output_height, &output_width, dilation, num_groups, bias);

    if (result != NULL)
    {
        // Print the result
        printf("Output Height: %d, Output Width: %d\n", output_height, output_width);
        for (int f = 0; f < num_filters; f++)
        {
            printf("Filter %d:\n", f);
            for (int i = 0; i < output_height; i++)
            {
                for (int j = 0; j < output_width; j++)
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
