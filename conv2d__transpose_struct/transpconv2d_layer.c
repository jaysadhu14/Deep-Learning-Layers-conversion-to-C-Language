#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "transpconv2d_config.h"
typedef float float32;
#include "hifi_model_trained_hdr.h"

int main()
{
    int dilation_values[2] = {1, 1};
    int stride_values[2] = {1, 1};

    transpconv2d_config *config = malloc(sizeof(transpconv2d_config));
    config->num_groups = 1;

    int image_height = 5;
    int image_width = 5;
    int image_channel = 8;

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

    config->image_channels = sizeof(filter[0][0][0]) / sizeof(filter[0][0][0][0]);
    config->kernel_height = sizeof(filter) / sizeof(filter[0]);
    config->kernel_width = sizeof(filter[0]) / sizeof(filter[0][0]);
    config->num_filters = sizeof(filter[0][0]) / sizeof(filter[0][0][0]);

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
        config->bias[i] = transpconv2d_bias[i];
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

    config->filter = malloc(config->kernel_height * sizeof(float ***));
    for (int i = 0; i < config->kernel_height; i++)
    {
        config->filter[i] = malloc(config->kernel_width * sizeof(float **));
        for (int j = 0; j < config->kernel_width; j++)
        {
            config->filter[i][j] = malloc(config->num_filters * sizeof(float *));
            for (int n = 0; n < config->num_filters; n++)
            {
                config->filter[i][j][n] = malloc(config->image_channels * sizeof(float));
                for (int c = 0; c < config->image_channels; c++)
                {
                    config->filter[i][j][n][c] = filter[i][j][n][c];
                }
            }
        }
    }

    
    int output_height, output_width ;

    float ***output = transpconv2d_execution(image, image_height, image_width, image_channel,
                                             config->filter, config->num_filters, config->kernel_height, config->kernel_width, &output_height, &output_width, config->bias,
                                             config->num_groups, config->stride, config->dilation);
    
    // Printing the result
    for (int i = 0; i < output_height; i++)
    {
        for (int j = 0; j < output_width; j++)
        {
            for (int f = 0; f < config->num_filters; f++)
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
            for (int f = 0; f < config->num_filters; f++)
            {
                free(config->filter[i][j][f]);
            }
            free(config->filter[i][j]);
        }
        free(config->filter[i]);
    }
    free(config->filter);
    
    return 0;
}
