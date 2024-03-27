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
    float ***padded_image = (float ***)calloc(sizeof(float **), image_channels);
    if (padded_image == NULL)
    {
        printf("Error: Memory allocation failed for padded image.\n");
        return NULL;
    }
    for (int c = 0; c < image_channels; c++)
    {
        padded_image[c] = (float **)calloc(sizeof(float *), padded_height);
        if (padded_image[c] == NULL)
        {
            printf("Error: Memory allocation failed for padded image.\n");
            return NULL;
        }
        for (int i = 0; i < padded_height; i++)
        {
            padded_image[c][i] = (float *)calloc(sizeof(float), padded_width);
            if (padded_image[c][i] == NULL)
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
                        padded_image[c][i][j] = 0.0; // Pad with zeros
                    }
                    else
                    {
                        padded_image[c][i][j] = image[c][original_row][original_col];
                    }
                }
                else if (strcmp(padding_mode, "valid") == 0)
                {
                    padded_image[c][i][j] = image[c][i][j];
                }
            }
        }
    }

    int new_kernel_height = (kernel_height - 1) * dilation[0] + 1;
    int new_kernel_width = (kernel_width - 1) * dilation[1] + 1;
    if (new_kernel_height <= 0 || new_kernel_width <= 0)
    {
        printf("Error: Calculated kernel dimensions are invalid.\n");
        return NULL;
    }

    // Allocate memory for dilated filters
    float ****dilated_filters = (float ****)calloc(sizeof(float ***), num_filters);
    if (dilated_filters == NULL)
    {
        printf("Error: Memory allocation failed for dilated filter.\n");
        return NULL;
    }
    for (int f = 0; f < num_filters; f++)
    {
        dilated_filters[f] = (float ***)calloc(sizeof(float **), image_channels);
        if (dilated_filters[f] == NULL)
        {
            printf("Error: Memory allocation failed for dilated filter.\n");
            return NULL;
        }

        for (int c = 0; c < image_channels; c++)
        {
            dilated_filters[f][c] = (float **)calloc(sizeof(float *), new_kernel_height);
            if (dilated_filters[f][c] == NULL)
            {
                printf("Error: Memory allocation failed for dilated filter.\n");
                return NULL;
            }
            for (int i = 0; i < new_kernel_height; i++)
            {
                dilated_filters[f][c][i] = (float *)calloc(sizeof(float), new_kernel_width);
                if (dilated_filters[f][c][i] == NULL)
                {
                    printf("Error: Memory allocation failed for dilated filter.\n");
                    return NULL;
                }
            }
        }
    }

    // Populate the dilated filters with values from the original filters
    for (int f = 0; f < num_filters; f++)
    {
        for (int c = 0; c < image_channels; c++)
        {
            for (int i = 0; i < kernel_height; i++)
            {
                for (int j = 0; j < kernel_width; j++)
                {
                    int dilated_i = i * dilation[0];
                    int dilated_j = j * dilation[1];
                    if (dilated_i < new_kernel_height && dilated_j < new_kernel_width)
                    {
                        dilated_filters[f][c][dilated_i][dilated_j] = filters[f][c][i][j];
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

    // Allocate memory for the result
    float ***result = (float ***)calloc(sizeof(float **), num_filters);
    if (result == NULL)
    {
        printf("Error: Memory allocation failed for result array.\n");
        return NULL;
    }
    for (int f = 0; f < num_filters; f++)
    {
        result[f] = (float **)calloc(sizeof(float *), *output_height);
        if (result[f] == NULL)
        {
            printf("Error: Memory allocation failed for result array.\n");
            return NULL;
        }
        for (int i = 0; i < *output_height; i++)
        {
            result[f][i] = (float *)calloc(sizeof(float), *output_width);
            if (result[f][i] == NULL)
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

    else if (new_kernel_height > image_height || new_kernel_width > image_width)
    {
        printf("Error: Filter size exceeds image size.\n");
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
                                    sum += padded_image[c][padded_i][padded_j] * dilated_filters[f][c][ki][kj];
                                }
                            }
                        }
                    }
                    sum += bias;
                    result[f][i][j] = sum;
                }
            }
        }
    }

    // Free memory allocated for padded_image
    for (int c = 0; c < image_channels; c++)
    {
        for (int i = 0; i < padded_height; i++)
        {
            free(padded_image[c][i]);
        }
        free(padded_image[c]);
    }
    free(padded_image);

    // Free memory for dilated filters
    for (int f = 0; f < num_filters; f++)
    {
        for (int c = 0; c < image_channels; c++)
        {
            for (int i = 0; i < new_kernel_height; i++)
            {
                free(dilated_filters[f][c][i]);
            }
            free(dilated_filters[f][c]);
        }
        free(dilated_filters[f]);
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
    if(image_height <= 0 || image_width <= 0)
    {
      printf("Error: invalid image size.\n");
            return -1;
    }
    float ***image = (float ***)malloc(image_channels * sizeof(float **));
    for (int c = 0; c < image_channels; c++)
    {
        image[c] = (float **)malloc(image_height * sizeof(float *));
        for (int i = 0; i < image_height; i++)
        {
            image[c][i] = (float *)malloc(image_width * sizeof(float));
        }
    }

    // Initialize image values
    float image_values[4][5][5] = {
        {{4, 3, 2, 3, 3},
         {2, 0, 1, 0, 1},
         {5, 2, 0, 1, 2},
         {3, 1, 2, 0, 4},
         {2, 3, 4, 1, 5}},
        {{4, 3, 2, 3, 3},
         {2, 0, 1, 0, 1},
         {5, 2, 0, 1, 2},
         {3, 1, 2, 0, 4},
         {2, 3, 4, 1, 5}},
        {{4, 3, 2, 3, 3},
         {2, 0, 1, 0, 1},
         {5, 2, 0, 1, 2},
         {3, 1, 2, 0, 4},
         {2, 3, 4, 1, 5}},
        {{4, 3, 2, 3, 3},
         {2, 0, 1, 0, 1},
         {5, 2, 0, 1, 2},
         {3, 1, 2, 0, 4},
         {2, 3, 4, 1, 5}}};

    for (int c = 0; c < image_channels; c++)
    {
        for (int i = 0; i < image_height; i++)
        {
            for (int j = 0; j < image_width; j++)
            {
                image[c][i][j] = image_values[c][i][j];
            }
        }
    }

    // Define filters
    int num_filters = 4;
    int kernel_channels = 4;
    int kernel_height = 3;
    int kernel_width = 3;
    if(kernel_height <= 0 || kernel_width <= 0)
    {
      printf("Error: invalid kernel size.\n");
            return -1;
    }
    if (image_channels != kernel_channels) {
        printf("Error: Image and kernel must have the same number of channels.\n");
        return -1; 
    }

    float ****filters = (float ****)malloc(num_filters * sizeof(float ***));
    for (int f = 0; f < num_filters; f++)
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

    // Initialize filter values
    float filter_values[4][4][3][3] = {
        {{{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}}},

        {{{0, 0, 0},
          {0, 0, 0},
          {0, 0, 0}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}}},

        {{{-1, 0, -2},
          {-2, 0, -2},
          {0, -3, -1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}}},

        {{{1, 1, 1},
          {1, 1, 1},
          {1, 1, 1}},
         {{1, 1, 1},
          {1, 1, 1},
          {1, 1, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}},
         {{1, 0, 2},
          {2, 0, 2},
          {0, 3, 1}}}};

    for (int f = 0; f < num_filters; f++)
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
    float bias = 0.5;

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
                    printf("%.2f ", result[f][i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // Free memory
    for (int c = 0; c < image_channels; c++)
    {
        for (int i = 0; i < image_height; i++)
        {
            free(image[c][i]);
        }
        free(image[c]);
    }
    free(image);

    for (int f = 0; f < num_filters; f++)
    {
        for (int c = 0; c < image_channels; c++)
        {
            for (int i = 0; i < kernel_height; i++)
            {
                free(filters[f][c][i]);
            }
            free(filters[f][c]);
        }
        free(filters[f]);
    }
    free(filters);

    for (int f = 0; f < num_filters; f++)
    {
        for (int i = 0; i < output_height; i++)
        {
            free(result[f][i]);
        }
        free(result[f]);
    }
    free(result);

    return 0;
}