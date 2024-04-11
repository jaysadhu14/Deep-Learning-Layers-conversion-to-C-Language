#pragma once

typedef struct
{
  float ****filter;
  float ***image;
  int image_channels;
  int kernel_height;
  int kernel_width;
  int num_filters;
  int *dilation;
  int *stride;
  float* bias;
  int num_groups;
} transpconv2d_config;


float ****kernel_dilation(float ****filters, int *kernel_height, int *kernel_width, int image_channels, int num_filters, int *dilation);
float*** transpconv2d_execution(float ***image, int image_height, int image_width, int image_channels,
                     float ****filters, int num_filters, int kernel_height, int kernel_width, 
                     int output_height, int output_width, float* bias,
                     int num_groups, int *stride,int* dilation);