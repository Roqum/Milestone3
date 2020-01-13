#include "channel.h"


Channel::Channel(size_t channels, size_t depth)
{
    channels_count = channels;
    bias = (((double) rand() / (RAND_MAX)) * 2 - 1) * 1/sqrt(channels *27);
    for(size_t i = 0; i<depth; i++){
        channel.push_back(Kernel(channels));
    }
}
