#ifndef CHANNEL_H
#define CHANNEL_H

#include <QCoreApplication>
#include <kernel.h>
#include <random>
/* fede's changes */
using namespace std;
typedef vector<double> vector1D;
typedef vector<vector1D> vector2D;
typedef vector<vector2D> vector3D;
/* end fede's changes */

class Channel
{
    /* fede's changes */
    vector3D activation_map;
    vector3D image;
    /* end fede's changes */
    vector<Kernel> channel;
    size_t channels_count;
    double bias;
public:
    Channel(size_t channels,  size_t depth);
    double leakyReLu();
    /* fede changes */
    double activation_map_get(size_t, size_t, size_t);
    void activation_map_set(size_t, size_t, size_t, double);
    size_t activation_map_size(){ return activation_map.size();}
    void image_set(size_t, size_t, size_t, double);
    /* end fede changes */
};

#endif // CHANNEL_H
