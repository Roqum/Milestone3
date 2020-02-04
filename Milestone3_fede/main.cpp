#include "cnn.h"
#include "kernel.h"
#include "neuron.h"
#include <iostream>
#include <channel.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));
    QApplication a(argc, argv);
    CNN w;
    /*
    vector<Channel> v_channels;
    for(unsigned int i = 0; i < 64; i++){
        v_channels.push_back(Channel(64,28));
    }
    */
    // stuff for backProp

    vector<Neuron> layer;
    for(unsigned int i = 0; i < 8000; i++){

        double randomValue = static_cast<double>(rand()) / RAND_MAX;
        layer.push_back(Neuron(randomValue));

        // layer.push_back(Neuron(i)); for debug
    }

    // end stuff for backProp
    w.maxpool3D_32();
    w.backProp_maxpool3D(layer);
    w.show();
    return a.exec();
}

