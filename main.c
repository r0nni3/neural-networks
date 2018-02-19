#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define INPUT_LAYER_SIZE 2
#define INPUT_LAYER_NODE_INPUT_SIZE 1

#define HIDDEN_LAYER_SIZE 3
#define HIDDEN_LAYER_NODE_INPUT_SIZE INPUT_LAYER_SIZE

#define OUTPUT_LAYER_SIZE 1
#define OUTPUT_LAYER_NODE_INPUT_SIZE HIDDEN_LAYER_SIZE

#define LAYER_NAME_MAX_LENGTH 140

/*
 *  Activation Functions
 */
double sigmoid (double in)
{
        return 1.0 / (1.0 + exp(-in));
}

double sigmoid_derivative (double sig)
{
        return sig * (1.0 - sig);
}

/*
 *  Neural Network Node
 */
struct NNNode {
        int input_size;
        double value;
        double bias;
        double *in;
        double *weight;
};

void init_node (struct NNNode *node, int input_size) {
        node->value = 0.0;
        node->bias = 0.0;
        node->input_size = input_size;
        node->in = calloc(input_size, sizeof(double));
        node->weight = calloc(input_size, sizeof(double));
}

void free_node (struct NNNode *node) {
        free(node->in);
        free(node->weight);
}

/*
 *  Neural Network Layer
 */
struct NNLayer {
        int size;
        char *name;
        struct NNNode *nodes;
};

void init_layer (struct NNLayer *layer, int size, int node_input_size, char *name)
{
        layer->size = size;
        layer->name = calloc(LAYER_NAME_MAX_LENGTH, sizeof(char));
        strcpy(layer->name, name);
        layer->nodes = calloc(size, sizeof(struct NNNode));
        for (int i = 0; i < size; ++i) {
                init_node(&layer->nodes[i], node_input_size);

        }
}

void free_layer (struct NNLayer *layer)
{
        for(int i = 0; i < layer->size; ++i) {
                free_node(&layer->nodes[i]);
        }
        free(layer->nodes);
        free(layer->name);
}

void print_layer (struct NNLayer *layer) {
        printf("%-11s %-12s\n", "=== Layer:", layer->name);
        for(int i = 0; i < layer->size; ++i) {
                printf("%-11s=== Node %02d\n", "", i+1);
                printf("%-15s%-9s %d\n", "", "Size:", layer->nodes[i].input_size);
                printf("%-15s%-9s %f\n", "", "Value:", layer->nodes[i].value);
                printf("%-15s%-9s %f\n", "", "Bias:", layer->nodes[i].bias);
                printf("%-15s%-9s %s", "", "IN:", "[");
                for(int j=0; j < layer->nodes[i].input_size; ++j) {
                        printf("%f", layer->nodes[i].in[j]);
                        if (j != layer->nodes[i].input_size -1) printf(", ");
                        else printf("]\n");
                }
                printf("%-15s%-9s %s", "", "WEIGHT:", "[");
                for(int j=0; j < layer->nodes[i].input_size; ++j) {
                        printf("%f", layer->nodes[i].weight[j]);
                        if (j != layer->nodes[i].input_size -1) printf(", ");
                        else printf("]\n");
                }
                printf("\n");
        }
        printf("\n");
}

void activate_layer (struct NNLayer *current, struct NNLayer *next) {
        for (int i = 0; i < next->size; ++i) {
                double sum = next->nodes[i].bias;
                for (int j = 0; j < current->size; ++j) {
                        double mult = current->nodes[j].value * next->nodes[i].weight[j];
                        next->nodes[i].in[j] = mult;
                        sum += mult;
                }
                next->nodes[i].value = sigmoid(sum);
        }
}

/*
 *  Input Layer
 */
void init_input_layer (struct NNLayer *layer)
{
        // Generic layer initialization
        init_layer(layer, INPUT_LAYER_SIZE, INPUT_LAYER_NODE_INPUT_SIZE, "Input");
        // First layer initialization
        for(int i = 0; i < layer->size; ++i) {
                layer->nodes[i].value = 1.0 * ((rand() % INPUT_LAYER_SIZE) + 1);
                layer->nodes[i].bias = 0.1 * ((rand() % INPUT_LAYER_SIZE) +1);
                for(int j = 0; j < layer->nodes[i].input_size; ++j) {
                        layer->nodes[i].in[j] = layer->nodes[i].value;
                        layer->nodes[i].weight[j] = 1.0;
                }
        }
}

void free_input_layer (struct NNLayer *layer)
{
        free_layer(layer);
}

/*
 *  Hidden Layer
 */
void init_hidden_layer (struct NNLayer *layer)
{
        init_layer(layer, HIDDEN_LAYER_SIZE, HIDDEN_LAYER_NODE_INPUT_SIZE, "Hidden");
        // First layer initialization
        for(int i = 0; i < layer->size; ++i) {
                for(int j = 0; j < layer->nodes[i].input_size; ++j) {
                        double b = rand() % (HIDDEN_LAYER_NODE_INPUT_SIZE);
                        double w = 0.1 * (b + 0.01);
                        layer->nodes[i].weight[j] = w;
                }
        }
}

void free_hidden_layer (struct NNLayer *layer)
{
        free_layer(layer);
}

/*
 *  Output Layer
 */
void init_output_layer (struct NNLayer *layer)
{
        init_layer(layer, OUTPUT_LAYER_SIZE, OUTPUT_LAYER_NODE_INPUT_SIZE, "Output");
        for(int i = 0; i < layer->size; ++i) {
                for(int j = 0; j < layer->nodes[i].input_size; ++j) {
                        double b = rand() % (OUTPUT_LAYER_NODE_INPUT_SIZE);
                        double w = 0.1 * (b + 0.01);
                        layer->nodes[i].weight[j] = w;
                }
        }
}

void free_output_layer (struct NNLayer *layer)
{
        free_layer(layer);
}

int main()
{
        struct NNLayer input_layer, hidden_layer, output_layer;
        srand(time(NULL));

        init_input_layer(&input_layer);
        init_hidden_layer(&hidden_layer);
        init_output_layer(&output_layer);

        activate_layer(&input_layer, &hidden_layer);
        activate_layer(&hidden_layer, &output_layer);

        print_layer(&input_layer);
        print_layer(&hidden_layer);
        print_layer(&output_layer);

        free_input_layer(&input_layer);
        free_hidden_layer(&hidden_layer);
        free_output_layer(&output_layer);

        return 0;
}
