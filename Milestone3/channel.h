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
    /* fede's changes */
    vector3D activation_map;
    vector3D image;
    /* end fede's changes */
public:
    Channel(size_t channels,  size_t depth);
    double leakyReLu(double);

    vector<Kernel> getChannel(){
        return channel;
    }
    Kernel getKernel(size_t index){
        return channel[index];
    }
    double getBias(){
        return bias;
    }
    /* fede changes */
    double activation_map_get(size_t, size_t, size_t);
    void activation_map_set(size_t, size_t, size_t, double);
    size_t activation_map_size(){ return activation_map.size();}
    void image_set(size_t, size_t, size_t, double);
    /* end fede changes */
};

#endif // CHANNEL_H
