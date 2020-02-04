#include "kernel.h"
#include <iostream>

Kernel::Kernel(size_t channel_amount)
{
    channels = channel_amount;
    kernel.resize(3);
    for(size_t i = 0; i<3;i++){
        kernel[i].resize(3);
        for(size_t j = 0; j<3;j++){
            kernel[i][j].resize(3);

        }
    }
    random_weights();
}
void Kernel::random_weights(){
    for(size_t i = 0; i<3;i++){
        for(size_t j = 0; j<3;j++){
            for(size_t x = 0; x<3;x++){
                kernel[i][j][x] = (((double) rand() / (RAND_MAX)) * 2 - 1) * 1/sqrt(channels *27);
                //cout << kernel[i][j][x] << endl;
            }
        }
    }
}

