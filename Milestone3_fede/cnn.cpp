#include "cnn.h"
#include "ui_cnn.h"
#include "iostream"
using namespace std;
#include<channel.h>
CNN::CNN(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CNN)
{
    conv3d_output.resize(32);
    for(unsigned int i = 0; i < 32; i++){
        conv3d_output[i].resize(20);
        for(unsigned int j = 0; j < 20; j++){
            conv3d_output[i][j].resize(20);
            for(unsigned int z = 0; z < 20; z++){
                conv3d_output[i][j][z].resize(20);
            }
        }
    }
    for(unsigned int i = 0; i < 32; i++){
        for(unsigned int j = 0; j < 20; j++){
            for(unsigned int z = 0; z < 20; z++){
                for(unsigned int k = 0; k < 20; k++){
                    conv3d_output[i][j][z][k] = j*20*20 + z*20 + k;
                }
            }
        }
    }
    image_32.resize(32);
    for(unsigned int i = 0; i < 32; i++){
        image_32[i].resize(10);
        for(unsigned int j = 0; j < 10; j++){
            image_32[i][j].resize(10);
            for(unsigned int z = 0; z < 10; z++){
                image_32[i][j][z].resize(10);
            }
        }
    }
    ui->setupUi(this);
}

CNN::~CNN()
{
    delete ui;
}

unsigned int CNN::vector_find_pos_max(vector<double> &values)
{
    double max = 0.0;
    unsigned int pos = 0;
    for(unsigned int i = 0; i < values.size(); i++){
        if(values[i] > max){
            max = values[i];
            pos = i;
        }
    }
    return pos;
}
void CNN::maxpool3D(vector<Channel> &v_channels)
{
    vector<double> values;
    values.resize(8);
    size_t v_channels_dim = v_channels.size(); // or 32 or 64 in this milestone
    size_t data_dim = v_channels[0].activation_map_size(); // or 20 or 10 in this milestone
    for(unsigned int i = 0; i < v_channels_dim; i++){


        for(unsigned int x = 0; x < data_dim / 2; x++){
            for(unsigned int y = 0; y < data_dim / 2; y++){
                for(unsigned int z = 0; z < data_dim / 2; z++){
                    // select the 8 values from which find the max

                    values[0] = v_channels[i].activation_map_get(x*2,y*2,z*2); // 0 0 0
                    values[1] = v_channels[i].activation_map_get(x*2,y*2,(z * 2) + 1); // 0 0 +1
                    values[2] = v_channels[i].activation_map_get(x*2,(y * 2) + 1,(z * 2)); // 0 +1 0
                    values[3] = v_channels[i].activation_map_get(x*2,(y * 2) + 1,(z * 2) + 1); // 0 +1 +1
                    values[4] = v_channels[i].activation_map_get((x * 2) + 1,y*2,z*2); // 1 0 0
                    values[5] = v_channels[i].activation_map_get((x * 2) + 1,y*2,(z * 2) + 1); // +1 0 +1
                    values[6] = v_channels[i].activation_map_get((x * 2) + 1,(y * 2) + 1,z*2); // +1 +1 0
                    values[7] = v_channels[i].activation_map_get((x * 2) + 1,(y * 2) + 1,(z * 2) + 1); // +1 +1 +1

                    unsigned int pos = vector_find_pos_max(values);
                    double max = values[pos];
                    // put 1 where the max is, 0 in the other pos

                    if(pos == 0)
                        v_channels[i].activation_map_set(x*2,y*2,z*2,1);
                    else
                        v_channels[i].activation_map_set(x*2,y*2,z*2,0);
                    if(pos == 1)
                        v_channels[i].activation_map_set(x*2,y*2,(z * 2) + 1,1);
                    else
                        v_channels[i].activation_map_set(x*2,y*2,(z * 2) + 1,0);
                    if(pos == 2)
                        v_channels[i].activation_map_set(x*2,(y * 2) + 1,z*2,1);
                    else
                        v_channels[i].activation_map_set(x*2,(y * 2) + 1,z*2,0);
                    if(pos == 3)
                        v_channels[i].activation_map_set(x*2,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        v_channels[i].activation_map_set(x*2,(y * 2) + 1,(z * 2) + 1,0);
                    if(pos == 4)
                        v_channels[i].activation_map_set((x * 2) + 1,y*2,z*2,1);
                    else
                        v_channels[i].activation_map_set((x * 2) + 1,y*2,z*2,0);
                    if(pos == 5)
                        v_channels[i].activation_map_set((x * 2) + 1,y*2,(z * 2) + 1,1);
                    else
                        v_channels[i].activation_map_set((x * 2) + 1,y*2,(z * 2) + 1,0);
                    if(pos == 6)
                        v_channels[i].activation_map_set((x * 2) + 1,(y * 2) + 1,z*2,1);
                    else
                        v_channels[i].activation_map_set((x * 2) + 1,(y * 2) + 1,z*2,0);
                    if(pos == 7)
                        v_channels[i].activation_map_set((x * 2) + 1,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        v_channels[i].activation_map_set((x * 2) + 1,(y * 2) + 1,(z * 2) + 1,0);

                    // add the max to the new matrix
                    v_channels[i].image_set(x,y,z,max);
                }
            }
        }


    }
}

/* fede changes */
double CNN::activation_map_get(vector3D& matrix, size_t x, size_t y, size_t z)
{
    return matrix[x][y][z];
}

void CNN::activation_map_set(vector3D& matrix, size_t x, size_t y, size_t z, double n)
{
    matrix[x][y][z] = n;
}

void CNN::image_set_32(vector3D& matrix, size_t x, size_t y, size_t z, double n)
{
    matrix[x][y][z] = n;
}
void CNN::image_set_64(vector3D& matrix, size_t x, size_t y, size_t z, double n)
{
    matrix[x][y][z] = n;
}
/* end fede changes */

void CNN::maxpool3D_32()
{
    vector<double> values;
    values.resize(8);
    size_t v_channels_dim = 32; // or 32 or 64 in this milestone
    size_t data_dim = 20; // or 20 or 10 in this milestone
    for(unsigned int i = 0; i < v_channels_dim; i++){


        for(unsigned int x = 0; x < data_dim / 2; x++){
            for(unsigned int y = 0; y < data_dim / 2; y++){
                for(unsigned int z = 0; z < data_dim / 2; z++){
                    // select the 8 values from which find the max

                    values[0] = activation_map_get(conv3d_output[i],x*2,y*2,z*2); // 0 0 0
                    values[1] = activation_map_get(conv3d_output[i],x*2,y*2,(z * 2) + 1); // 0 0 +1
                    values[2] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2)); // 0 +1 0
                    values[3] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1); // 0 +1 +1
                    values[4] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,z*2); // 1 0 0
                    values[5] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1); // +1 0 +1
                    values[6] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2); // +1 +1 0
                    values[7] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1); // +1 +1 +1

                    unsigned int pos = vector_find_pos_max(values);
                    double max = values[pos];
                    // put 1 where the max is, 0 in the other pos

                    if(pos == 0)
                        activation_map_set(conv3d_output[i],x*2,y*2,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,y*2,z*2,0);
                    if(pos == 1)
                        activation_map_set(conv3d_output[i],x*2,y*2,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,y*2,(z * 2) + 1,0);
                    if(pos == 2)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,z*2,0);
                    if(pos == 3)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1,0);
                    if(pos == 4)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,z*2,0);
                    if(pos == 5)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1,0);
                    if(pos == 6)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2,0);
                    if(pos == 7)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1,0);

                    // add the max to the new matrix
                    image_set_32(image_32[i],x,y,z,max);
                }
            }
        }


    }
}

void CNN::maxpool3D_64()
{
    vector<double> values;
    values.resize(8);
    size_t v_channels_dim = 64; // or 32 or 64 in this milestone
    size_t data_dim = 10; // or 20 or 10 in this milestone
    for(unsigned int i = 0; i < v_channels_dim; i++){


        for(unsigned int x = 0; x < data_dim / 2; x++){
            for(unsigned int y = 0; y < data_dim / 2; y++){
                for(unsigned int z = 0; z < data_dim / 2; z++){
                    // select the 8 values from which find the max

                    values[0] = activation_map_get(conv3d_output[i],x*2,y*2,z*2); // 0 0 0
                    values[1] = activation_map_get(conv3d_output[i],x*2,y*2,(z * 2) + 1); // 0 0 +1
                    values[2] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2)); // 0 +1 0
                    values[3] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1); // 0 +1 +1
                    values[4] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,z*2); // 1 0 0
                    values[5] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1); // +1 0 +1
                    values[6] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2); // +1 +1 0
                    values[7] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1); // +1 +1 +1

                    unsigned int pos = vector_find_pos_max(values);
                    double max = values[pos];
                    // put 1 where the max is, 0 in the other pos

                    if(pos == 0)
                        activation_map_set(conv3d_output[i],x*2,y*2,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,y*2,z*2,0);
                    if(pos == 1)
                        activation_map_set(conv3d_output[i],x*2,y*2,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,y*2,(z * 2) + 1,0);
                    if(pos == 2)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,z*2,0);
                    if(pos == 3)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1,0);
                    if(pos == 4)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,z*2,0);
                    if(pos == 5)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1,0);
                    if(pos == 6)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2,0);
                    if(pos == 7)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1,1);
                    else
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1,0);

                    // add the max to the new matrix
                    image_set_64(image_64[i],x,y,z,max);
                }
            }
        }


    }
}

void CNN::backProp_maxpool3D_64(vector<Neuron> &layer)
{
    vector<double> values;
    values.resize(8);
    unsigned int v_channel_dim = 64; // 32 or 64 in this milestone
    unsigned int data_dim = 10;

    for(unsigned int i = 0; i < v_channel_dim; i++){

        for(unsigned int x = 0; x < data_dim/2; x++){
            for(unsigned int y = 0; y < data_dim/2; y++){
                for(unsigned int z = 0; z < data_dim/2; z++){
                    // now activation map ha only 1 and 0 values
                    values[0] = activation_map_get(conv3d_output[i],x*2,y*2,z*2); // 0 0 0
                    values[1] = activation_map_get(conv3d_output[i],x*2,y*2,(z * 2) + 1); // 0 0 +1
                    values[2] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2)); // 0 +1 0
                    values[3] = activation_map_get(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1); // 0 +1 +1
                    values[4] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,z*2); // 1 0 0
                    values[5] = activation_map_get(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1); // +1 0 +1
                    values[6] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2); // +1 +1 0
                    values[7] = activation_map_get(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1); // +1 +1 +1

                    unsigned int linear_pos = x*data_dim*data_dim + y*data_dim + z;
                    std::cout << linear_pos << "\n" << std::endl;
                    double current_gradient = layer[linear_pos].get_Gradient();
                    // put gradient where there is 1
                    if(values[0] == 1.)
                        activation_map_set(conv3d_output[i],x*2,y*2,z*2,current_gradient);
                    else if(values[1] == 1.)
                        activation_map_set(conv3d_output[i],x*2,y*2,(z * 2) + 1,current_gradient);
                    else if(values[2] == 1.)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,z*2,current_gradient);
                    else if(values[3] == 1.)
                        activation_map_set(conv3d_output[i],x*2,(y * 2) + 1,(z * 2) + 1,current_gradient);
                    else if(values[4] == 1.)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,z*2,current_gradient);
                    else if(values[5] == 1.)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,y*2,(z * 2) + 1,current_gradient);
                    else if(values[6] == 1.)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,z*2,current_gradient);
                    else if(values[7] == 1.)
                        activation_map_set(conv3d_output[i],(x * 2) + 1,(y * 2) + 1,(z * 2) + 1,current_gradient);

                }
            }
        }


    }
}
