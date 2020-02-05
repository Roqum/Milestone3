#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include "neuron.h"
#include "cnn.h"
#include <QCoreApplication>
using namespace std;


typedef vector<Layer> Network;
typedef vector<size_t> topology;


class Neural_Network
{
public:
    double loss;

    // counters used for implement batches
    size_t batch_size;
    size_t in_batch_counter = 0;
    bool last_sample = false;
    bool batch_done = false;

    // used for import one Datafile
    string raw_input;
    vector <double> processed_input;

    topology net_topology; // topology of the network
    Network network;    // network as vector of layers

    /// constructor creates a neural network with given topology and batches ///
    Neural_Network(topology net_topology, size_t batch_size);

    /// calculates and returns the output of the neural network ///
    vector<long double> softmax_net_output(vector<long double> net_output);

    /// imports a datafile ///
    void import_data(string filename);

    /// converts the datafile to vector of numbers of the file ///
    void change_input();

    /// feed forward the imported datafile ///
    void calc_data();

    /// starts the backpropagation and optimizes the weights ///
    void backpropagation(vector <double> correct_output, int counter);

    /// calculates the loss for the graph ///
    void calc_loss_function(vector<double> t, vector<double> o);

    /// is analyzing the datafile ///
    vector<double> analyze_input();

    /// gets the loss after a data runthrough ///
    double getloss();
};

#endif // NEURAL_NETWORK_H
