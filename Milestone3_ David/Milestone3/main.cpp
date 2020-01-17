#include "cnn.h"
#include "kernel.h"
#include <iostream>

#include <QApplication>

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));
    QApplication a(argc, argv);
    CNN w;
    Kernel kern(32);
    w.show();
    return a.exec();
}

