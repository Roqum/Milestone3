#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <neural_network.h>
#include <QMainWindow>
#include <QCoreApplication>
#include <QFileDialog>
#include <QDirIterator>
#include <QDir>


QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QFileDialog *data;

    /// trains the network with given settings of ///
    void run_training();

    /// starts the test runs ///
    void run_tests();

    /// start analyzing the datafile ///
    void run_input_analysis();

    // used for graphs
    QVector<double> x0,x1,x2,x3,x4,y0,y1,y2,y3,y4;


private:
    Ui::MainWindow *ui;

    Neural_Network *network;
    long double current_average = 0;

    int epoch;  // amount of datapoints on the graph
    int nFiles; // amount of data files for each epoch
    int nFiles_test;    // amount of files for test runs
    QDir nqgp;  // directory of nqgp files
    QDir qgp;   // directory of qgp files

public slots:
    /// gets directory of the test files ///
    void on_loadData_button_clicked();

    /// runs the data into the network ///
    void analyze_clicked();

private slots:

    /// creates the graphs ///
    void makePlot();

};
#endif // MAINWINDOW_H
