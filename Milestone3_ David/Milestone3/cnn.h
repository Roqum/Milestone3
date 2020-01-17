#ifndef CNN_H
#define CNN_H

#include <QMainWindow>
#include <channel.h>

QT_BEGIN_NAMESPACE
namespace Ui { class CNN; }
QT_END_NAMESPACE

class CNN : public QMainWindow
{
    Q_OBJECT
    vector<Channel> channel32;
    vector<Channel> channel64;
    typedef vector<vector<vector<double>>> matrix3d;
    vector<matrix3d> conv3d_output;
    vector<matrix3d> conv3d_input;

public:
    CNN(QWidget *parent = nullptr);
    ~CNN();
    void conv3D(vector<Channel> chan);
    //void conv3Dx64();
    void maxpool3D();

private:
    Ui::CNN *ui;
};
#endif // CNN_H
