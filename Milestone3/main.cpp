#include "cnn.h"
#include "kernel.h"
#include <iostream>
#include <channel.h>
#include <QApplication>

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));
    QApplication a(argc, argv);
    CNN w;
    vector<Channel> v_channels;
    for(unsigned int i = 0; i < 32; i++){
        v_channels.push_back(Channel(32,28));
    }
    w.maxpool3D(v_channels);
    w.show();
    return a.exec();
}

