#ifndef CHANNEL_H
#define CHANNEL_H

#include <QCoreApplication>
#include <kernel.h>
#include <random>

using namespace std;
typedef vector<double> vector1D;
typedef vector<vector1D> vector2D;
typedef vector<vector2D> vector3D;

class Channel
{
    vector<Kernel> channel;
    size_t channels_count;
    double bias;
public:

    Channel(size_t channels,  size_t depth);

    ///runs the value through the leaky ReLu function///
    double leakyReLu(double);

    ///returns the Channels///
    vector<Kernel> getChannel(){
        return channel;
    }

    ///returns the Kernel at the given index///
    Kernel getKernel(size_t index){
        return channel[index];
    }

    ///returns the bias of the channel///
    double getBias(){
        return bias;
    }
};

#endif // CHANNEL_H
