#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef float float32; // Define float32 as an alias for float
#include "hifi_model_trained_hdr.h" // Now include the header

float ***transposedconvolution(float ***image, int image_height, int image_width, int image_channels,
                               float ****filters, int num_filters, int kernel_height, int kernel_width,
                               int *stride, char *padding_mode, int *output_height, int *output_width, int *dilation, int num_groups, float* bias)
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

    int new_kernel_height = (kernel_height - 1) * dilation[0] + 1;
    int new_kernel_width = (kernel_width - 1) * dilation[1] + 1;
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
    for (int i = 0; i < kernel_height; i++)
    {
        for (int j = 0; j < kernel_width; j++)
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

    // Calculate padded dimensions
    if (strcmp(padding_mode, "same") == 0)
    {
        *output_height = ((image_height - 1) * stride[0]) + new_kernel_height - 1;
        *output_width = ((image_width - 1) * stride[1]) + new_kernel_width - 1;
    }
    else if (strcmp(padding_mode, "valid") == 0)
    {
        *output_height = ((image_height - 1) * stride[0]) + new_kernel_height;
        *output_width = ((image_width - 1) * stride[1]) + new_kernel_width;
    }
    else
    {
        printf("Invalid padding mode. Please use 'same' or 'valid'.\n");
        return NULL;
    }

    if (*output_height <= 0 || *output_width <= 0)
    {
        printf("Error: Calculated output dimensions are invalid.\n");
        return NULL;
    }

    // Allocate memory for the result
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
    int res_width = 0;
    int res_height = 0;

    // Perform convolution
    for (int group = 0; group < num_groups; group++)
    {
        int filter_start = group * group_num_filter;
        int filter_end = (group + 1) * group_num_filter;
        for (int f = filter_start; f < filter_end; f++)
        {
            for (int i = 0; i < image_height; i++)
            {
                for (int j = 0; j < image_width; j++)
                {
                    float temp = 0.0;
                    int group_kernel_channels = (image_channels / num_groups);
                    for (int c = 0; c < group_kernel_channels; c++)
                    {
                        for (int ki = 0; ki < new_kernel_height; ki++)
                        {
                            for (int kj = 0; kj < new_kernel_width; kj++)
                            {
                                if (strcmp(padding_mode, "same") == 0)
                                {
                                    temp = image[i][j][c] * dilated_filters[ki][kj][f][c];
                                    res_height = (stride[0] * i) + ki - 1;
                                    res_width = (stride[1] * j) + kj - 1;
                                    if (res_height != -1 && res_width != -1)
                                    {
                                        result[res_height][res_width][f] += temp;
                                    }
                                }
                                else if (strcmp(padding_mode, "valid") == 0)
                                {
                                    temp = image[i][j][c] * dilated_filters[ki][kj][f][c];
                                    res_height = (stride[0] * i) + ki;
                                    res_width = (stride[1] * j) + kj;
                                    result[res_height][res_width][f] += temp;
                                }
                            }
                        }
                    }

                    result[i][j][f] += bias[f];
                }
            }
        }
    }

    // Free memory for dilated filters
    for (int h = 0; h < new_kernel_height; h++)
    {
        for (int w = 0; w < new_kernel_width; w++)
        {
            for (int f = 0; f < num_filters; f++)
            {
                free(dilated_filters[h][w][f]);
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
    int image_channels = 8;

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

    // Initialize image values
    float image_values[5][5][8] =
    {{{0.47082111, 0.56454155, 0.47411139, 0.39789071, 0.44853204,
          0.73068485, 0.86399994, 0.07422767},
         {0.84931383, 0.85477148, 0.3568433 , 0.81971868, 0.2452446 ,
          0.17581495, 0.23532371, 0.82130231},
         {0.10966626, 0.67651353, 0.9491342 , 0.29720435, 0.24681756,
          0.41709466, 0.57499046, 0.23735115},
         {0.70311151, 0.14250128, 0.21326755, 0.90256225, 0.37859084,
          0.69157972, 0.33527991, 0.22003314},
         {0.320075  , 0.6328385 , 0.24901225, 0.65323447, 0.08193617,
          0.87434319, 0.20142911, 0.84109871}},

        {{0.43992733, 0.90050966, 0.71342332, 0.35606619, 0.42981462,
          0.80004913, 0.49894221, 0.88943446},
         {0.48347193, 0.11624301, 0.5034592 , 0.60121954, 0.93783206,
          0.90958671, 0.43012235, 0.7001503 },
         {0.26764639, 0.09322448, 0.04715708, 0.34360132, 0.30328439,
          0.78465257, 0.37492467, 0.39985707},
         {0.99590235, 0.0529896 , 0.54476707, 0.08893046, 0.31033404,
          0.49429855, 0.19977257, 0.63259171},
         {0.19787385, 0.77740652, 0.39865911, 0.45815434, 0.12908281,
          0.29792037, 0.25828668, 0.0247611 }},

        {{0.56382659, 0.3843806 , 0.90024501, 0.50966552, 0.25845184,
          0.86192758, 0.03573309, 0.44177938},
         {0.05342786, 0.85708581, 0.78502468, 0.40264566, 0.83456178,
          0.97971013, 0.33007688, 0.81740086},
         {0.82792315, 0.75205605, 0.99675929, 0.88993893, 0.28082819,
          0.81816243, 0.12611045, 0.17582594},
         {0.91708744, 0.34180377, 0.81655768, 0.70653926, 0.58907065,
          0.31909701, 0.77601887, 0.67886718},
         {0.81333257, 0.15588383, 0.00980667, 0.2263475 , 0.83212004,
          0.51246037, 0.96148066, 0.35771257}},

        {{0.1334777 , 0.18267404, 0.61158808, 0.21438524, 0.73119164,
          0.37309286, 0.24513713, 0.05494139},
         {0.97238197, 0.24038321, 0.38770631, 0.37276013, 0.09430313,
          0.36728164, 0.17581295, 0.40919958},
         {0.42191877, 0.01056206, 0.48821478, 0.61123547, 0.51830685,
          0.01179892, 0.65295177, 0.81663706},
         {0.81955934, 0.92119509, 0.3263054 , 0.09643456, 0.81602562,
          0.333474  , 0.88284411, 0.63278305},
         {0.81802024, 0.73195767, 0.0866748 , 0.32363475, 0.01440459,
          0.68355194, 0.22075446, 0.49370036}},

        {{0.85226599, 0.61282596, 0.19582962, 0.25271528, 0.34738061,
          0.28668712, 0.57674677, 0.56241291},
         {0.56442808, 0.67804174, 0.38969444, 0.97815744, 0.72092444,
          0.7808571 , 0.81811824, 0.72363202},
         {0.41234505, 0.42368865, 0.00230308, 0.47147632, 0.87512535,
          0.79093439, 0.99769394, 0.50612915},
         {0.29459411, 0.18518311, 0.03302318, 0.97702318, 0.7971422 ,
          0.55256733, 0.37147917, 0.55254005},
         {0.37472828, 0.74743972, 0.29964215, 0.22073714, 0.21230503,
          0.30943901, 0.10620103, 0.27147412}}};

    for (int i=0;i<image_height;i++)
    {
        for (int j=0;j<image_width;j++)
        {
            for (int c=0;c<image_channels;c++)
            {
                image[i][j][c] = image_values[i][j][c];
            }
        }
    }

    // Perform convolution with "valid" padding
    int kernel_channel =  sizeof(conv2d_transpose_1_kernel[0][0][0]) / sizeof(conv2d_transpose_1_kernel[0][0][0][0]);
    int kernel_height = sizeof(conv2d_transpose_1_kernel)/sizeof(conv2d_transpose_1_kernel[0]);
    int kernel_width = sizeof(conv2d_transpose_1_kernel[0]) / sizeof(conv2d_transpose_1_kernel[0][0]);
    int num_filters = sizeof(conv2d_transpose_1_kernel[0][0]) / sizeof(conv2d_transpose_1_kernel[0][0][0]);
    
    if (kernel_height <= 0 || kernel_width <= 0)
    {
        printf("Error: invalid kernel size.\n");
        return -1;
    }
    if (image_channels != kernel_channel)
    {
        printf("Error: Image and kernel must have the same number of channels.\n");
        return -1;
    }
   
    float ****filters = (float ****)malloc(kernel_height * sizeof(float ***));
    for (int h = 0; h < kernel_height; h++)
    {
        filters[h] = (float ***)malloc(kernel_width * sizeof(float **));
        for (int w = 0; w < kernel_width; w++)
        {
            filters[h][w] = (float **)malloc(num_filters * sizeof(float *));
            for (int c = 0; c < num_filters; c++)
            {
                filters[h][w][c] = (float *)malloc(image_channels * sizeof(float));
            }
        }
    }


    for (int h = 0; h < kernel_height; h++)
    {
        for (int w = 0; w < kernel_width; w++)
        {
            for (int f = 0; f < num_filters; f++)
            {
                for (int c = 0; c < image_channels; c++)
                {
                    filters[h][w][f][c] = conv2d_transpose_1_kernel[h][w][f][c];
                }
            }
        }
    }
  

    // Define stride and dilation
    int stride[2] = {1, 1};
    int dilation[2] = {1, 1};
    int num_groups = 1;
    float* bias = (float*)calloc(sizeof(float),num_filters);
    for(int i = 0;i<num_filters;i++)
    {
        bias[i] = conv2d_transpose_1_bias[i];
    }

    // Perform convolution with "valid" padding
    int output_height, output_width;
    float ***result = transposedconvolution(image, image_height, image_width, image_channels,
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

    // Free memory
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

    for (int h = 0; h < output_height; h++)
    {
        for (int w = 0; w < output_width; w++)
        {
            free(result[h][w]);
        }
        free(result[h]);
    }
    free(result);

    return 0;
 }