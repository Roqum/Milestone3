#include "cnn.h"
#include "ui_cnn.h"
using namespace std;
#include<channel.h>
CNN::CNN(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CNN)
{
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
                    /*
                    values[0] = v_channels[i].activation_map_get(x,y,z); // 0 0 0
                    values[1] = v_channels[i].activation_map_get(x,y,(z + 1)); // 0 0 +1
                    values[2] = v_channels[i].activation_map_get(x,(y + 1),z); // 0 +1 0
                    values[3] = v_channels[i].activation_map_get(x,(y + 1),(z + 1)); // 0 +1 +1
                    values[4] = v_channels[i].activation_map_get((x + 1),y,z); // 1 0 0
                    values[5] = v_channels[i].activation_map_get((x + 1),y,(z + 1)); // +1 0 +1
                    values[6] = v_channels[i].activation_map_get((x + 1),(y + 1),z); // +1 +1 0
                    values[7] = v_channels[i].activation_map_get((x + 1),(y + 1),(z + 1)); // +1 +1 +1
                    */
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

                    /*
                    if(pos == 0)
                        v_channels[i].activation_map_set(x,y,z,1);
                    else
                        v_channels[i].activation_map_set(x,y,z,0);
                    if(pos == 1)
                        v_channels[i].activation_map_set(x,y,(z + 1),1);
                    else
                        v_channels[i].activation_map_set(x,y,(z + 1),0);
                    if(pos == 2)
                        v_channels[i].activation_map_set(x,(y + 1),z,1);
                    else
                        v_channels[i].activation_map_set(x,(y + 1),z,0);
                    if(pos == 3)
                        v_channels[i].activation_map_set(x,(y + 1),(z + 1),1);
                    else
                        v_channels[i].activation_map_set(x,(y + 1),(z + 1),0);
                    if(pos == 4)
                        v_channels[i].activation_map_set((x + 1),y,z,1);
                    else
                        v_channels[i].activation_map_set((x + 1),y,z,0);
                    if(pos == 5)
                        v_channels[i].activation_map_set((x + 1),y,(z + 1) ,1);
                    else
                        v_channels[i].activation_map_set((x + 1),y,(z + 1) ,0);
                    if(pos == 6)
                        v_channels[i].activation_map_set((x + 1),(y + 1),z,1);
                    else
                        v_channels[i].activation_map_set((x + 1),(y + 1),z,0);
                    if(pos == 7)
                        v_channels[i].activation_map_set((x + 1),(y + 1),(z + 1),1);
                    else
                        v_channels[i].activation_map_set((x + 1),(y + 1),(z + 1),0);
                    */
                    // add the max to the new matrix
                    v_channels[i].image_set(x,y,z,max);
                }
            }
        }


    }
}
