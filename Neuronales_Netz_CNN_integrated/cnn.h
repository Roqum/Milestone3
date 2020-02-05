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
    //general input/output
    vector<matrix3d> conv3d_output;
    vector<matrix3d> conv3d_input;
    vector<matrix3d> cnn_output;
    vector<matrix3d> cnn_input;
    vector<matrix3d> matrix4d;

    //raw input
    vector<double> processed_input;
    string raw_input;

    //conv3d
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

    double activation_map_get(vector3D&, size_t, size_t, size_t);
    void activation_map_set(vector3D&,size_t, size_t, size_t, double);
    size_t activation_map_size(){ return cnn_output.size();}
    void image_set_32(vector3D&, size_t, size_t, size_t, double);
    void image_set_64(vector3D&, size_t, size_t, size_t, double);

    //paddin and input data
    void padding();
    vector<double> flatten_matrix(vector<vector<vector<vector<double>>>> input_matrix);
    vector<vector<vector<vector<double>>>> rebuild_input_matrix(vector<double> input);
    void change_input();
    void import_data(string filename);

    vector<vector3D> get_cnn_matrix();
    vector<double> get_processed_input();
    //set and get
    vector<Channel> getConv3DLayer32();
    vector<Channel> getConv3DLayer64();


private:
};
#endif // CNN_H

