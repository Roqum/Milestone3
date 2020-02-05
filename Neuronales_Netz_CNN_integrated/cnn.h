#ifndef CNN_H
#define CNN_H

#include <QMainWindow>
#include <channel.h>
#include "neuron.h"
#include <iostream>
#include <fstream>
using namespace std;
typedef vector<vector<vector<double>>> matrix3d;

class CNN
{
    //input/output for conv3d operation
    vector<vector3D> conv3d_output;
    vector<vector3D> conv3d_input;
    vector<matrix3d> matrix4d;

    //raw input of the files
    vector<double> processed_input;
    string raw_input;

    //vector of channels for conv3d
    vector<Channel> channel32;
    vector<Channel> channel64;

    //maxpool
    vector3D activation_map;
    vector<vector3D> image_32;
    vector<vector3D> image_64;
public:
    CNN();
    void clear_all();

    ///starts the feed forward for the CNN///
    vector<double> feed_forward();

    //conv3D stuff
    ///runs the conv3d operation///
    void conv3D(vector<Channel> chan);

    //maxpool stuff
    ///runs the maxpool operation for 32 inputs///
    void maxpool3D_32();

    ///runs the maxpool operation for 64 inputs///
    void maxpool3D_64();

    ///runs the backpropagation for maxpool 32///
    void backProp_maxpool3D_64(vector<Neuron> &);

    ///runs the backpropagation for maxpool 64///
    void backProp_maxpool3D_32(vector<Neuron> &);

    ///find the position of the maximum value of an vector///
    unsigned int vector_find_pos_max(vector<double> &);

    ///gets the value of a 3D vector at the given index///
    double activation_map_get(vector3D&, size_t, size_t, size_t);

    ///sets the value in a 3D vector at the given index///
    void activation_map_set(vector3D&,size_t, size_t, size_t, double);

    ///returns the size of the conv3D operation output///
    size_t activation_map_size(){ return conv3d_output.size();}

    //paddin and input data

    ///add the zeros arround the 3 Dimensional matrixes in the conv3d Input///
    void padding();

    ///flattens the given 4D Matrix///
    vector<double> flatten_matrix(vector<vector<vector<vector<double>>>> input_matrix);

    ///build a 4D matrix from a input string///
    vector<vector<vector<vector<double>>>> rebuild_input_matrix(vector<double> input);


    void change_input();

    ///imports a file///
    void import_data(string filename);

    ///returns the conv3d input list of 3D matrixes///
    vector<vector3D> get_cnn_matrix();

    ///returns the string of doubles///
    vector<double> get_processed_input();

    ///returns the 32 Channels///
    vector<Channel> getConv3DLayer32();

    ///returns the 64 Channels///
    vector<Channel> getConv3DLayer64();


private:
};
#endif // CNN_H

