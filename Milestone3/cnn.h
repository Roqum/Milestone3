#ifndef CNN_H
#define CNN_H

#include <QMainWindow>
#include <channel.h>

class CNN
{
    vector<Channel> channel32;
    vector<Channel> channel64;
    typedef vector<vector<vector<double>>> matrix3d;
    vector<matrix3d> conv3d_output;
    vector<matrix3d> conv3d_input;
    vector<matrix3d> matrix4d;

    vector<double> processed_input;
public:
    CNN();
    ~CNN();
    void conv3D(vector<Channel> chan);
    //void conv3Dx64();
    void maxpool3D(vector<Channel> &);
    unsigned int vector_find_pos_max(vector<double>&);

    //paddin and input data
    void padding();
    vector<double> flatten_matrix(vector<vector<vector<vector<double>>>> input_matrix);
    vector<vector<vector<vector<double>>>> rebuild_input_matrix(vector<double> input);

private:
};
#endif // CNN_H
