#include "cnn.h"
#include "ui_cnn.h"

CNN::CNN(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CNN)
{
    for(size_t i=0;i<32;i++){
        channel32.push_back(Channel(32,28));
    }
    for(size_t i=0;i<64;i++){
        channel64.push_back(Channel(64,28));
    }

    ui->setupUi(this);
}

CNN::~CNN()
{
    delete ui;
}

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
                            conv3d_output[i][m-1][x-1][y-1] = chan[i].leakyReLu(kernel_sum);
                        }
                }
            }
        }


    //goes through each Kernel of all channels
    /*for(size_t i = 0; i< chan.size(); i++){
        for(size_t j = 0; j< chan.size(); j++){
            chan[i].getKernel(j)(1,1,1);
        }
    }*/
}
//void CNN::conv3Dx32()


