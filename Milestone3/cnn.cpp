#include "cnn.h"
#include "ui_cnn.h"

CNN::CNN(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::CNN)
{
    ui->setupUi(this);
}

CNN::~CNN()
{
    delete ui;
}

