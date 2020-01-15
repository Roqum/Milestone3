/********************************************************************************
** Form generated from reading UI file 'cnn.ui'
**
** Created by: Qt User Interface Compiler version 5.13.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CNN_H
#define UI_CNN_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_CNN
{
public:
    QWidget *centralwidget;
    QMenuBar *menubar;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *CNN)
    {
        if (CNN->objectName().isEmpty())
            CNN->setObjectName(QString::fromUtf8("CNN"));
        CNN->resize(800, 600);
        centralwidget = new QWidget(CNN);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        CNN->setCentralWidget(centralwidget);
        menubar = new QMenuBar(CNN);
        menubar->setObjectName(QString::fromUtf8("menubar"));
        CNN->setMenuBar(menubar);
        statusbar = new QStatusBar(CNN);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        CNN->setStatusBar(statusbar);

        retranslateUi(CNN);

        QMetaObject::connectSlotsByName(CNN);
    } // setupUi

    void retranslateUi(QMainWindow *CNN)
    {
        CNN->setWindowTitle(QCoreApplication::translate("CNN", "CNN", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CNN: public Ui_CNN {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CNN_H
