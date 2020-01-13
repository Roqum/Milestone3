#ifndef CHANNEL_H
#define CHANNEL_H

#include <QCoreApplication>
#include <kernel.h>
#include <random>

class Channel
{
    vector<Kernel> channel;
    size_t channels_count;
    double bias;
public:
    Channel(size_t channels,  size_t depth);
    double leakyReLu();
};

#endif // CHANNEL_H
