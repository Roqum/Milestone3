#include "mainwindow.h"
#include "neuron.h"
#include "neural_network.h"
#include <QApplication>
#include <iostream>

int main(int argc, char *argv[])
{
    srand(static_cast<unsigned int>(time(nullptr)));

    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}
