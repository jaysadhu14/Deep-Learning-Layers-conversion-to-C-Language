// // -----------------------------------------------  FINAL LOGIC CODE -----------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef float float32;              // Define float32 as an alias for float
#include "hifi_model_trained_hdr.h" // Now include the header

// Sigmoid activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Tanh activation function
double tanh_activation(double x)
{
    return tanh(x);
}

void print(int n, double layer[])
{
    for (int i = 0; i < n; i++)
    {
        printf("%.17g \n", layer[i]);
    }
    printf("\n");
}

void transpose_and_concatenate(double **input, double **output, int input_size, int unit)
{
    for (int k = 0; k < 3; k++)
    {
        for (int i = 0; i < unit; i++)
        {
            for (int j = 0; j < input_size; j++)
            {
                int transposed_row = (k * input_size) + j;
                int transposed_col = (k * unit) + i;

                output[i][transposed_row] = input[j][transposed_col];
            }
        }
    }

   
}

void dense(int id, int n, int m, double *input, double **weight, double **bias, double output[], int weight_value, int unit)
{
    for (int i = 0; i < n; ++i)
    {
        output[i] = 0;
        for (int j = (0 + (m * id)); j < (m + (m * id)); ++j)
        {
            output[i] += (weight[i][j] * input[j - (m * id)]);
          
        }
        // Apply activation function if needed
        // (activation function code goes here)
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = (0 + (unit * id)); j < (unit + (unit * id)); ++j)
        {
           output[i] += bias[weight_value][j];
        }
    }
}

// Forward pass of GRU for each layer
double **gru(int unit, int num_layers, int input_size, double **input, double *hidden_prev,
             double **transposed_kernel, double **hidden_prev_weight,
             double **bias)
{
    int id, weight_value;
    // Allocate memory for storing each layer's hidden state
    double **layer_outputs = (double **)malloc(num_layers * sizeof(double *));
    if (layer_outputs == NULL)
    {
        printf("Memory allocation failed.\n");
        return NULL;
    }

    // Forward pass for each layer
    for (int layer = 0; layer < num_layers; layer++)
    {
        // Allocate memory for the hidden state of the current layer
        layer_outputs[layer] = (double *)malloc(unit * sizeof(double));
        if (layer_outputs[layer] == NULL)
        {
            printf("Memory allocation failed.\n");
            return NULL;
        }

        // -------------------------------- UPDATE GATE ------------------------------------

        id = 0;

        double *input_dense1 = (double *)malloc(unit * sizeof(double));
        weight_value = 0;
        dense(id, unit, input_size, input[layer], transposed_kernel, bias, input_dense1, weight_value, unit);

        double *hidden_prev_dense1 = (double *)malloc(unit * sizeof(double));
        weight_value = 1;
        dense(id, unit, unit, hidden_prev, hidden_prev_weight, bias, hidden_prev_dense1, weight_value, unit);

        double *update_gate = (double *)malloc(unit * sizeof(double));
        for (int i = 0; i < unit; i++)
        {
            update_gate[i] = input_dense1[i] + hidden_prev_dense1[i];
            update_gate[i] = sigmoid(update_gate[i]);
        }

        // -------------------------------- RESET GATE ------------------------------------

        id = 1;

        double *input_dense2 = (double *)malloc(unit * sizeof(double));
        weight_value = 0;
        dense(id, unit, input_size, input[layer], transposed_kernel, bias, input_dense2, weight_value, unit);

        double *hidden_prev_dense2 = (double *)malloc(unit * sizeof(double));
        weight_value = 1;
        dense(id, unit, unit, hidden_prev, hidden_prev_weight, bias, hidden_prev_dense2, weight_value, unit);

        double *reset_gate = (double *)malloc(unit * sizeof(double));
        for (int i = 0; i < unit; i++)
        {
            reset_gate[i] = input_dense2[i] + hidden_prev_dense2[i];
            reset_gate[i] = sigmoid(reset_gate[i]);
        }

        // -------------------------------- CANDIDATE HIDDEN -------------------------------

        id = 2;

        double *input_dense3 = (double *)malloc(unit * sizeof(double));
        weight_value = 0;
        dense(id, unit, input_size, input[layer], transposed_kernel, bias, input_dense3, weight_value, unit);

        double *hidden_prev_dense3 = (double *)malloc(unit * sizeof(double));
        weight_value = 1;
        dense(id, unit, unit, hidden_prev, hidden_prev_weight, bias, hidden_prev_dense3, weight_value, unit);

        double *candidate_hidden = (double *)malloc(unit * sizeof(double));
        for (int i = 0; i < unit; i++)
        {
            candidate_hidden[i] = input_dense3[i] + (reset_gate[i] * hidden_prev_dense3[i]);
            candidate_hidden[i] = tanh_activation(candidate_hidden[i]);
        }

        // -------------------------------- HIDDEN ----------------------------------------

        for (int i = 0; i < unit; i++)
        {
            hidden_prev[i] = (update_gate[i] * hidden_prev[i]) + ((1 - update_gate[i]) * candidate_hidden[i]);
        }

        // Store the hidden state of the current layer
        for (int i = 0; i < unit; i++)
        {
            layer_outputs[layer][i] = hidden_prev[i];
        }

        // Free dynamically allocated memory for the current layer
        free(input_dense1);
        free(hidden_prev_dense1);
        free(reset_gate);
        free(input_dense2);
        free(hidden_prev_dense2);
        free(update_gate);
        free(input_dense3);
        free(hidden_prev_dense3);
        free(candidate_hidden);
    }

    return layer_outputs;
}

int main()
{
    int num_layers = 1;
    double hidden_prev_value[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double input_values[1][64] = {{0.47637218, 0.47232874, 0.10006439, 0.65960998, 0.18673663,
         0.87878256, 0.66238111, 0.76815995, 0.51716311, 0.70981192,
         0.36708417, 0.68924672, 0.3011454 , 0.77961593, 0.25662221,
         0.04709273, 0.61420269, 0.59986929, 0.15164045, 0.0170851 ,
         0.95081352, 0.0607439 , 0.6265779 , 0.30833318, 0.33380683,
         0.18424628, 0.90719183, 0.52202839, 0.62564159, 0.21200452,
         0.81683269, 0.54351541, 0.48566905, 0.88709811, 0.76747826,
         0.36386286, 0.19028927, 0.25362354, 0.52663637, 0.87948085,
         0.97126507, 0.59546867, 0.89165427, 0.57032343, 0.11570343,
         0.12485452, 0.71210728, 0.48673068, 0.96335789, 0.87646222,
         0.68542523, 0.13255757, 0.72568856, 0.91785024, 0.27703435,
         0.89551934, 0.35075227, 0.89713249, 0.9727304 , 0.22751713,
         0.97903469, 0.37426406, 0.03749561, 0.31970003}};
    int input_size = sizeof(gru_kernel) / sizeof(gru_kernel[0]);
    int unit = sizeof(gru_recurrent_kernel) / sizeof(gru_recurrent_kernel[0]);
    int unit_3 = unit * 3;

    double **input = (double **)malloc(num_layers * sizeof(double *));
    if (input == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }
    for (int i = 0; i < num_layers; i++)
    {
        input[i] = (double *)malloc(input_size * sizeof(double));
        if (input[i] == NULL)
        {
            printf("Memory allocation failed.\n");
            return 1;
        }
    }
    for (int i = 0; i < num_layers; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            input[i][j] = input_values[i][j];
        }
    }
    double *hidden_prev = (double *)malloc(unit * sizeof(double));
    if (hidden_prev == NULL)
    {
        printf("Memory allocation failed.\n");
        return 1;
    }
    for (int i = 0; i < unit; i++)
    {
        hidden_prev[i] = hidden_prev_value[i];
    }
    double **bias = (double **)malloc(2 * sizeof(double *));
    for (int i = 0; i < 2; i++)
    {
        bias[i] = (double *)malloc((unit * 3) * sizeof(double));
    }

    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < (unit * 3); j++)
        {
            bias[i][j] = gru_bias[i][j];
        }
    }
    double **transposed_kernel = (double **)calloc(sizeof(double *), unit);
    int input_sizes = 3 * input_size;
    for (int i = 0; i < unit; i++)
    {
        transposed_kernel[i] = (double *)calloc(sizeof(double), input_sizes);
    }
    double **input_weight = (double **)calloc(sizeof(double *), input_size);
    for (int i = 0; i < input_size; i++)
    {
        input_weight[i] = (double *)calloc(sizeof(double), unit_3);
    }
    for (int i = 0; i < input_size; i++)
    {
        for (int j = 0; j < unit_3; j++)
        {
            input_weight[i][j] = gru_kernel[i][j];
        }
    }

    transpose_and_concatenate(input_weight, transposed_kernel, input_size, unit);

    double **transposed_recurrent = (double **)calloc(sizeof(double *), unit);
    double **hidden_prev_weights = (double **)calloc(sizeof(double *), unit);
    for (int i = 0; i < unit; i++)
    {
        transposed_recurrent[i] = (double *)calloc(sizeof(double), unit_3);
        hidden_prev_weights[i] = (double *)calloc(sizeof(double), unit_3);
    }
    for (int i = 0; i < unit; i++)
    {
        for (int j = 0; j < unit_3; j++)
        {
            hidden_prev_weights[i][j] = gru_recurrent_kernel[i][j];
        }
    }
    transpose_and_concatenate(hidden_prev_weights, transposed_recurrent, unit, unit);

    double **layer_outputs = gru(unit, num_layers, input_size, input, hidden_prev, transposed_kernel, transposed_recurrent, bias);

    if (layer_outputs != NULL)
    {
        printf("num_layer - %d\n", num_layers);
        printf("unit - %d\n", unit);
        // Print layer outputs
        for (int layer = 0; layer < num_layers; layer++)
        {
            printf("Layer %d output:\n", layer);
            print(unit, layer_outputs[layer]);
        }
    }
    else
    {
        printf("Error: Unable to perform forward pass.\n");
    }
        // Free dynamically allocated memory
        for (int i = 0; i < num_layers; i++)
        {
            free(layer_outputs[i]);
            free(input[i]);
        }

        free(layer_outputs);
        free(input);
        free(hidden_prev);

        for (int i = 0; i < unit; i++)
        {
            free(transposed_kernel[i]);
            free(transposed_recurrent[i]);
        }

        free(transposed_kernel);
        free(transposed_recurrent);
    
    
    return 0;
}