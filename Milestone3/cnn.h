#ifndef CNN_H
#define CNN_H

#include <QMainWindow>

#include<channel.h>
QT_BEGIN_NAMESPACE
namespace Ui { class CNN; }
QT_END_NAMESPACE

class CNN : public QMainWindow
{
    Q_OBJECT

public:
    CNN(QWidget *parent = nullptr);
    ~CNN();
    void conv3D();
    void maxpool3D(vector<Channel> &);
    unsigned int vector_find_pos_max(vector<double>&);

private:
    Ui::CNN *ui;
};
#endif // CNN_H
