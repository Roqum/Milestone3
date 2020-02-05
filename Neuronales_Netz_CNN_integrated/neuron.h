#ifndef NEURON_H
#define NEURON_H

#include <QCoreApplication>

using namespace std;

typedef vector<float> Vector;
class Neuron;
typedef vector<Neuron> Layer;


class Neuron
{
    Layer *pre_layer;       // saves the previous layer
    size_t layer_index;     // saves the neurons index of his layer
    long double input_sum, output;  // saves the incoming and outcoming value

    //Variables for ADAM
    long double gradient;
    long double bias_m, bias_v;
    long double m;
    long double v;
    const double b1 = 0.9;
    const double b2 = 0.999;
    const double epsilon = 1e-8;
    const double learning_rate = 0.0005;

    // gradients from neurons of the next layer
    vector<long double> next_layer_gradients;

    // saves the outgoing weights of this neuron
    vector<long double> weights;

public:
    //saves the type of neuron
    enum type {Input, Hidden, Output} neuron_type;

    /// contructor builds a neuron with given parameters ///
    Neuron( Layer *prev_layer, type typ, int topology, size_t layer_index);

    /// runs the LReLu function with the summed up input ///
    void calc_LReLu_output();

    /// returns the derivative of the LReLu function ///
    double LReLu_derivative();

    /// sums up all incoming values of the neuron ///
    void calc_Input();

    /// calculates the gradient with target value for the output Layer ///
    void calc_gradient(double target_value, size_t batch_size);

    /// calculates the gradient without target value for hidden Layers ///
    void calc_gradient(size_t batch_size);

    /// adjusts the new weights ///
    void correct_weights(int counter);

    /// correct the bias ///
    void correct_bias(int counter);


    /*** set- and get functions ***/

    /// returns the outgoing weight with the given index ///
    long double getWeights(size_t index);

    /// set the weight with the given index to the given value ///
    void setWeights(size_t neuron_indx, long double weight);

    /// sets the output to the given value ///
    void setOutput(long double input);

    /// returns the input of the neuron ///
    long double getInput();

    /// sets the input ///
    double setInput(double inp);

    /// returns the output of the neuron ///
    long double getOutput();

    /// sets the gradient to the given value ///
    void setGradient(double new_gradient);

    ///gets the gradient of the neuron///
    double get_Gradient();


};

#endif // NEURON_H
