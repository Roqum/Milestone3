#include "cnn.h"     //test

/*CNN::CNN()
{
    for(size_t i=0;i<32;i++){
        channel32.push_back(Channel(32,28));
    }
    for(size_t i=0;i<64;i++){
        channel64.push_back(Channel(64,32));
    }
}*/

CNN::CNN()

{
    for(size_t i=0;i<32;i++){
        channel32.push_back(Channel(32,28));
    }
    for(size_t i=0;i<64;i++){
        channel64.push_back(Channel(64,28));
    }


}

/*CNN::~CNN()
{

}*/

void CNN::conv3D(vector<Channel> chan){
    double kernel_sum = 0;
    //goes through the input matrix
        for (size_t m = 1; m<conv3d_input[0].size()-1;m++){
            for (size_t x = 1; x<conv3d_input[0][0].size()-1;x++){
                for (size_t y = 1; y<conv3d_input[0][0][0].size()-1;y++)
                {

                    // goes through each channel
                    for(size_t i = 0; i< chan.size(); i++){
                        kernel_sum = 0;
                            for(size_t j = 0; j< chan.size(); j++){
                                // takes the sum of kernel * input_matrix operation
                                kernel_sum +=
                                        chan[i].getKernel(j)(0,0,0)*conv3d_input[j][m-1][x-1][y-1] +
                                        chan[i].getKernel(j)(1,0,0)*conv3d_input[j][m]  [x-1][y-1] +
                                        chan[i].getKernel(j)(2,0,0)*conv3d_input[j][m+1][x-1][y-1] +
                                        chan[i].getKernel(j)(0,1,0)*conv3d_input[j][m-1][x]  [y-1] +
                                        chan[i].getKernel(j)(0,2,0)*conv3d_input[j][m-1][x+1][y-1] +
                                        chan[i].getKernel(j)(0,0,1)*conv3d_input[j][m-1][x-1][y] +
                                        chan[i].getKernel(j)(0,0,2)*conv3d_input[j][m-1][x-1][y+1] +

                                        chan[i].getKernel(j)(1,1,0)*conv3d_input[j][m]  [x]  [y-1] +
                                        chan[i].getKernel(j)(2,1,0)*conv3d_input[j][m+1][x]  [y-1] +
                                        chan[i].getKernel(j)(2,2,0)*conv3d_input[j][m+1][x+1][y-1] +
                                        chan[i].getKernel(j)(1,2,0)*conv3d_input[j][m]  [x+1][y-1] +

                                        chan[i].getKernel(j)(0,1,1)*conv3d_input[j][m-1][x]  [y] +
                                        chan[i].getKernel(j)(0,2,1)*conv3d_input[j][m-1][x+1][y] +
                                        chan[i].getKernel(j)(0,2,2)*conv3d_input[j][m-1][x+1][y+1] +
                                        chan[i].getKernel(j)(0,1,2)*conv3d_input[j][m-1][x][y+1] +

                                        chan[i].getKernel(j)(1,0,1)*conv3d_input[j][m]  [x-1][y] +
                                        chan[i].getKernel(j)(2,0,1)*conv3d_input[j][m+1][x-1][y] +
                                        chan[i].getKernel(j)(2,0,2)*conv3d_input[j][m+1][x-1][y+1] +
                                        chan[i].getKernel(j)(1,0,2)*conv3d_input[j][m]  [x-1][y+1] +

                                        chan[i].getKernel(j)(1,1,1)*conv3d_input[j][m]  [x]  [y] +

                                        chan[i].getKernel(j)(2,1,1)*conv3d_input[j][m+1][x]  [y] +
                                        chan[i].getKernel(j)(1,2,1)*conv3d_input[j][m]  [x+1][y] +
                                        chan[i].getKernel(j)(1,1,2)*conv3d_input[j][m]  [x]  [y+1] +
                                        chan[i].getKernel(j)(2,2,1)*conv3d_input[j][m+1][x+1][y] +
                                        chan[i].getKernel(j)(2,1,2)*conv3d_input[j][m+1][x]  [y+1] +
                                        chan[i].getKernel(j)(1,2,2)*conv3d_input[j][m][x+1]  [y+1] +

                                        chan[i].getKernel(j)(2,2,2)*conv3d_input[j][m+1][x+1][y+1];
                            }
                            kernel_sum += chan[i].getBias();
                            //saves the output of each cell
                            //conv3d_output[i][m-1][x-1][y-1] = chan[i].leakyReLu(kernel_sum);      //test LRelu not implemented
                        }
                }
            }
        }
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

//Padding and input-data
vector<vector<vector<vector<double>>>> CNN::rebuild_input_matrix(vector<double> input){
    size_t p_ind=0, mom_ind=0, phi_ind=0, theta_ind=0;
    vector<vector<vector<vector<double>>>> input_matrix(28, vector<vector<vector<double>>>(20, vector<vector<double>>(20, vector<double>(20))));

    for (size_t i=0; i<input.size();i++){
       input_matrix[p_ind][mom_ind][phi_ind][theta_ind] = input[i];
       theta_ind ++;
       if (theta_ind ==20){
           theta_ind=0;
           phi_ind++;
           if (phi_ind==20){
               phi_ind =0;
               mom_ind++;
               if (mom_ind==20){
                   mom_ind=0;
                   p_ind++;
               }
           }
       }
    }
    conv3d_input = input_matrix;
    return input_matrix;
}
vector<double> CNN::flatten_matrix(vector<vector<vector<vector<double>>>> input_matrix){
    vector<double> flattened_matrix;
    for (size_t p_ind=0;p_ind<28;p_ind++){
        for (size_t mom_ind=0;mom_ind<28;mom_ind++){
            for (size_t phi_ind=0;phi_ind<28;phi_ind++){
                for (size_t theta_ind=0;theta_ind<28;theta_ind++){
                    flattened_matrix.push_back(input_matrix[p_ind][mom_ind][phi_ind][theta_ind]);
                }
            }
        }
    }
    processed_input = flattened_matrix;
    return flattened_matrix;
}

void CNN::padding(){
    //size_t p_ind=0, mom_ind=0, phi_ind=0, theta_ind=0;
    vector<double> zero_vector;
    vector<vector<double>> zero_matrix;     //3d padding
    for (size_t p_ind =0;p_ind< conv3d_input.size();p_ind++){
        for (size_t mom_ind =0;mom_ind< conv3d_input [p_ind].size();mom_ind++){
            for (size_t phi_ind =0;phi_ind< conv3d_input [p_ind] [mom_ind].size();phi_ind++){
                conv3d_input [p_ind] [mom_ind] [phi_ind].insert(conv3d_input[p_ind] [mom_ind] [phi_ind].begin(),0);
                conv3d_input [p_ind] [mom_ind] [phi_ind].push_back(0);
                zero_vector.push_back(0);
            }
            conv3d_input [p_ind] [mom_ind].insert(conv3d_input[p_ind] [mom_ind].begin(),zero_vector);
            conv3d_input [p_ind] [mom_ind].insert(conv3d_input[p_ind] [mom_ind].end(),zero_vector);
            zero_matrix.push_back(zero_vector);
            zero_vector.clear();
        }
        conv3d_input [p_ind].insert(conv3d_input[p_ind].begin(),zero_matrix);
        conv3d_input [p_ind].insert(conv3d_input[p_ind].end(),zero_matrix);
        zero_matrix.clear();
    }
}

void CNN::import_data(string filename){
    raw_input.clear();
    string store_data_line;
    ifstream myFile(filename); // opens file
    if(!myFile.is_open()){
        cout << "Was not able to open the file.\n";
        exit(1);
    }
    // writes the whole file in one string raw_input
    while(myFile.eof() == false){
        getline(myFile, store_data_line);
        raw_input += store_data_line + " ";
    }
    myFile.close();
}


void CNN::change_input(){
    processed_input.clear();
    string number_helper="";

    // adds each number in raw_input to a vector
    for(size_t i=0; i<raw_input.length();i++){
        if(isdigit(raw_input[i])){
            number_helper += raw_input[i];
        }
        else if (number_helper != ""){
            processed_input.push_back(static_cast <double> (stoi(number_helper)));//nur int werte wegen stoi()
            number_helper="";
        }
    }
    raw_input.clear();
}
//void CNN::conv3Dx32()
vector<vector3D> CNN::get_cnn_matrix(){
    return conv3d_input;
}

vector<double> CNN::get_processed_input(){
    return processed_input;
}


