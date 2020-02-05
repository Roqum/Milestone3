#include "channel.h"


Channel::Channel(size_t channels, size_t depth)
{
    channels_count = channels;

    //sets random bias for the channel
    bias = (((double) rand() / (RAND_MAX)) * 2 - 1) * 1/sqrt(channels *27);

    //creates kernels in amount of the depth
    for(size_t i = 0; i<depth; i++){
        channel.push_back(Kernel(channels));
    }
}

//leakyReLu function
double Channel::leakyReLu(double inp){
    if (inp < 0){
        return (0.01*inp);
    }
    else {
        return inp;
    }
}

