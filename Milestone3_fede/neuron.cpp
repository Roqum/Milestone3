#include "neuron.h"

Neuron::Neuron(double gradient)
{
    this->gradient = gradient;
}

double Neuron::get_Gradient()
{
    return this->gradient;
}
