#include "channel.h"


Channel::Channel(size_t channels, size_t depth)
{
    channels_count = channels;
    bias = (((double) rand() / (RAND_MAX)) * 2 - 1) * 1/sqrt(channels *27);
    for(size_t i = 0; i<depth; i++){
        channel.push_back(Kernel(channels));
        /* fede changes */
        unsigned int dim_data;
        // creating space in the activation map
        if(channels == 32)
            dim_data = 20;
        else
            dim_data = 10;
        activation_map.resize(dim_data);
        for(unsigned int i = 0; i < activation_map_size(); i++){
            activation_map[i].resize(activation_map_size());
            for(unsigned int j = 0; j < activation_map_size(); j++){
                activation_map[i][j].resize(activation_map_size());
            }
        }
        // initialising activation map, just for debugging
        for(unsigned int x = 0; x < dim_data; x++){
            for(unsigned int y = 0; y < dim_data; y++){
                for(unsigned int z = 0; z < dim_data; z++){
                    activation_map[x][y][z] = static_cast<double>(rand()) / RAND_MAX;
                    //activation_map[x][y][z] = x*dim_data*dim_data + y*dim_data + z; for super debugging
                }
            }
        }
        // creating space in the result image
        image.resize(activation_map_size()/2);
        for(unsigned int i = 0; i < image.size(); i++){
            image[i].resize(image.size());
            for(unsigned int j = 0; j < image.size(); j++){
                image[i][j].resize(image.size());
            }
        }
        /* end fede changes */
    }
}

/* fede changes */
double Channel::activation_map_get(size_t x, size_t y, size_t z)
{
    return activation_map[x][y][z];
}

void Channel::activation_map_set(size_t x, size_t y, size_t z, double n)
{
    activation_map[x][y][z] = n;
}

void Channel::image_set(size_t x, size_t y, size_t z, double n)
{
    image[x][y][z] = n;
}
/* end fede changes */

double Channel::leakyReLu(double inp){
    if (inp < 0){
        return (0.01*inp);
    }
    else {
        return inp;
    }
}
