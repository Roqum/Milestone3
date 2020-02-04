#ifndef CNN_H
#define CNN_H

#include <QMainWindow>
#include "neuron.h"
#include<channel.h>
typedef vector<double> vector1D;
typedef vector<vector1D> vector2D;
typedef vector<vector2D> vector3D;
QT_BEGIN_NAMESPACE
namespace Ui { class CNN; }
QT_END_NAMESPACE

class CNN : public QMainWindow
{
    Q_OBJECT
    vector<vector3D> conv3d_output;
    vector<vector3D> image_32;
    vector<vector3D> image_64;

public:
    CNN(QWidget *parent = nullptr);
    ~CNN();
    void conv3D();
    void maxpool3D(vector<Channel> &);
    void maxpool3D_32();
    void maxpool3D_64();
    void backProp_maxpool3D_64(vector<Neuron> &);
    void backProp_maxpool3D_32(vector<Neuron> &);
    unsigned int vector_find_pos_max(vector<double> &);
    /* fede changes */
    double activation_map_get(vector3D&, size_t, size_t, size_t);
    void activation_map_set(vector3D&,size_t, size_t, size_t, double);
    size_t activation_map_size(){ return conv3d_output.size();}
    void image_set_32(vector3D&, size_t, size_t, size_t, double);
    void image_set_64(vector3D&, size_t, size_t, size_t, double);
    /* end fede changes */

private:
    Ui::CNN *ui;
};
#endif // CNN_H
