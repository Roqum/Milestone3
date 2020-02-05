#include "neuron.h"
#include <bits/stdc++.h>

Neuron::Neuron(Layer *prev_layer, type typ, int topology, size_t layer_indx) // topology=number of neurons next layer
{
    bias_m = 0;
    bias_v = 0;
    m = 0;
    v = 0;
    gradient = 0;

    // avaoid error because input layer have no previous layer
    if (prev_layer != nullptr){
        pre_layer = prev_layer;
    }

    // gradients of layer+1 are saved here later
    next_layer_gradients.resize(topology);

    layer_index = layer_indx; // index of this neuron in his layer
    neuron_type = typ; // saves neuron type

    // sets random weights for each weight
    for (size_t i = 0; i < topology; i++){
        weights.push_back((sqrt((2/ (long double) 5000)) * ((long double)rand() / (long double)RAND_MAX )));
    }
}

void Neuron::calc_Input(){
    vector<long double> input_values;

    // saves all output values of previous neurons multiplied by the weights to this neuron in one vector
    for(size_t i = 0; i < pre_layer->size(); i++){
        input_values.push_back((*pre_layer)[i].output * (*pre_layer)[i].weights[layer_index]);
    }
    long double sum = 0;
    // sums up all values times their weights
    for(size_t i = 0; i < input_values.size(); i++){
        sum += input_values[i];
    }
    input_sum = sum; // input of the neuron is saved
    input_values.clear();
}

void Neuron::calc_LReLu_output(){
    // runs LReLu on the input value of the neuron
    if (input_sum < 0){
        output = input_sum * 0.01;
    }
    else output = input_sum;

}

void Neuron::calc_gradient(double target_value, size_t batch_size){

    if (neuron_type == Output){
        // calculates gradient of the output neuron
        gradient += ((target_value - output )* (output * (1-output)))/batch_size;

        // push backs its gradient to all previous neurons
        for (size_t i = 0; i < pre_layer->size(); i++){
            (*pre_layer)[i].next_layer_gradients[layer_index] = gradient;
        }
    }
    else{
        cout << "Error!";
        exit(1);
    }

}
void Neuron::calc_gradient(size_t batch_size){

    if (neuron_type == Hidden){
        long double sum = 0;
        // sums up all gradients which were push backed
        for (size_t i = 0; i < next_layer_gradients.size(); i++){
            sum += next_layer_gradients[i] * weights[i];
        }     
        sum = sum * LReLu_derivative(); // sum is multiplied by LReLu derivative
        gradient += sum/batch_size;

        // push backs all gradients of this neuron to neurons of previous layer
        for (size_t i = 0; i < pre_layer->size(); i++){
            (*pre_layer)[i].next_layer_gradients[layer_index] = gradient;
        }

    }
    // calculates gradient of Input neuron same as for hidden layer
    if (neuron_type == Input){
        long double sum = 0;
        for (size_t i = 0; i < next_layer_gradients.size(); i++){
            sum += next_layer_gradients[i] * weights[i];
        }
        sum = sum * LReLu_derivative();
        gradient = sum/batch_size;
    }

}

double Neuron::LReLu_derivative(){
    if (input_sum >= 0){
        return 1;
    }
    else return 0.01;
}

void Neuron::correct_weights(int counter){
    correct_bias(counter); // corrects the bias
    if(neuron_type != Input){
        // corrects all incoming weights with this formular
        // new_weight =old_weight + (η/(sprt(bias_v)+ϵ))* bias_m
        for (size_t i = 0; i<pre_layer->size(); i++){
            (*pre_layer)[i].setWeights(layer_index,((*pre_layer)[i].getWeights(layer_index) + ((learning_rate)/(sqrt(bias_v) + epsilon))* bias_m));
        }
    }
}
void Neuron::correct_bias(int counter){
    // all biases are updated
    m = (b1*m)+((1-b1)*(gradient));
    v = (b2*v)+((1-b2)*pow(gradient,2));
    bias_m = m /(1- pow(b1,counter));
    bias_v = v /(1- pow(b2,counter));
}



/******* SET- AND GET-FUNCTIONS *******/

long double Neuron::getWeights(size_t index){
    return this->weights[index];
}
void Neuron::setWeights(size_t neuron_indx, long double weight){
    this->weights[neuron_indx] = weight;
}
void Neuron::setOutput(long double inp){
    output = inp;
}
long double Neuron::getInput(){
    return input_sum;
}
double Neuron::setInput(double inp){
    input_sum = inp;
}
long double Neuron::getOutput(){
    return output;
}
void Neuron::setGradient(double new_gradient){
    gradient = new_gradient;
}
double Neuron::get_Gradient()
{
    return this->gradient;
}
