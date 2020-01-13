#ifndef CNN_H
#define CNN_H

#include <QMainWindow>

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
    void maxpool3D();

private:
    Ui::CNN *ui;
};
#endif // CNN_H
