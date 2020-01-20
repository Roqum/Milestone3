#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "neural_network.h"
#include <iostream>

using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    makePlot();     // creates the graphs

    // conncects the buttons with a function
    connect(ui->load_data, SIGNAL (released()), this, SLOT (on_loadData_button_clicked()));
    connect(ui->analyze, SIGNAL (released()), this, SLOT (analyze_clicked()));

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_loadData_button_clicked()
{

    //opens loading file window
    data = new QFileDialog(this);
    data->setAcceptMode(QFileDialog::AcceptOpen);

    //QString line;
    //string line_s;
    QString directory;

    //open the file which is selected by the user
    directory = data->QFileDialog::getExistingDirectory();
    QDirIterator dir_iter(directory,QDir::AllEntries | QDir::NoDotAndDotDot);
    nqgp = dir_iter.next();
    qgp = dir_iter.next();

    vector<double> inputVals;
    vector<double> targetVals;
    if(nFiles < 0){
        QMessageBox messageBox;
        messageBox.setText("error for the param, must be positive!!!");
        messageBox.exec();
        return;
    }

}

void MainWindow::analyze_clicked(){

    // gets the value of amount of data per epoch, number of tests and number of epochs
    nFiles = ui->training_lineEdit->text().toInt();
    nFiles_test = ui->testing_lineEdit->text().toInt();
    epoch = ui->spinBox->value();

    // sets up graph ranges
    ui->widget->xAxis->setRange(0,epoch);
    ui->widget_2->xAxis->setRange(0,epoch);

    // creates a network with 0,1 or 2 hidden Layers
    // choice depens on selceted network mode
    if( ui->comboBox->currentIndex() == 2){
        topology topo{224000,64,32,2};
        network = new Neural_Network(topo, ui->batches->value());
    }
    else if ( ui->comboBox->currentIndex() == 1){
        topology topo{224000,64,2};
        network = new Neural_Network(topo, ui->batches->value());
    }
    else {
        topology topo{224000,2};
        network = new Neural_Network(topo, ui->batches->value());
    }

    // runs in given graph mode
    if (ui->comboBox_2->currentIndex() ==0){
        run_training();     // starts the training
        run_tests();        // runs tests
    }
    else if(ui->comboBox_2->currentIndex() ==1){
        run_input_analysis(); // run network to analyze the data
    }
}

void MainWindow::run_training(){

    vector<double> result;
    vector<double> nqpg_value{0,1};
    vector<double> qpg_value{1,0};

    // saves a datafile
    QString fileAddress;
    QString fileNames;

    // counter to select training files from the directory
    int nqgp_iter = 0;
    int qgp_iter = 0;

    // counts the run throughs
    int cnt = 0;

    // runs nFiles files for each epoch
    for (int e = 0; e<epoch;e++){
        for(int i = 0; i < nFiles; i++){

            // if counter is even the network gets a nqpg file
            // else he gets a qpg file
            if(cnt % 2 == 0){

                // opens a nqgp datafile
                fileAddress = QString::fromStdString("/phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." + to_string(nqgp_iter) + "_event.dat");
                fileNames = nqgp.path() + fileAddress;

                nqgp_iter++;
                nqgp_iter = nqgp_iter % 100; // the first 100 files are just used for training

                // runs the data through the network
                network->import_data(fileNames.toStdString());
                network->change_input();
                network->calc_data();

                // starts the backpropagation
                network->backpropagation(nqpg_value, cnt+1);

                // gets the output of the network to calculate the loss function
                result.push_back(network->network[network->network.size()-1][0].getOutput());
                result.push_back(network->network[network->network.size()-1][1].getOutput());
                network->calc_loss_function(nqpg_value,result);

                current_average += result[1];   // used for graph
                result.clear();
            }

            // if counter is even the network gets a qpg file
            else {
                //opens a qgp datafile
                fileAddress = QString::fromStdString("/phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." + to_string(qgp_iter) + "_event.dat");
                fileNames = qgp.path() + fileAddress;

                qgp_iter++;
                qgp_iter = qgp_iter % 100; // the first 100 files are just used for training

                // runs the data through the network
                network->import_data(fileNames.toStdString());
                network->change_input();
                network->calc_data();

                // starts backpropagation
                network->backpropagation(qpg_value, cnt+1);

                // gets the output of the network to calculate the loss function
                result.push_back(network->network[network->network.size()-1][0].getOutput());
                result.push_back(network->network[network->network.size()-1][1].getOutput());
                network->calc_loss_function(qpg_value,result);

                current_average += result[0];   //used for graph
                result.clear();
            }

            // checks if its the last file
            if(i==nFiles-1){
                network->last_sample = true;
            }

            cnt++;

            //prints the output and loss in the console
            /*cout << endl << "Training: " << cnt <<  endl;
            cout<< "qgp: " <<network->network[network->network.size()-1][0].getOutput() << endl;
            cout<< "nqgp: " <<network->network[network->network.size()-1][1].getOutput() << endl;
            cout<< "loss-function: " << network->getloss() << endl;*/

            QApplication::processEvents(); // so the window does not freeze
        }

        //graphic for the average
        current_average /= nFiles;
        double size = x0.size();
        x0.push_back(size);
        y0.push_back(current_average * 100);
        current_average = 0;
        //graphic for the loss function
        size = x1.size();
        x1.push_back(size);
        y1.push_back(network->getloss());

        // pass data points to graphs:
        ui->widget->graph(0)->setData(x0, y0);
        ui->widget_2->graph(0)->setData(x1, y1);
        ui->widget->replot();
        ui->widget_2->replot();

        qApp->processEvents(); // avoids window freezing
    }


}


void MainWindow::run_tests(){
    vector<long double> result;

    // saves datafile
    QString fileAddress;
    QString fileNames;

    int cnt = 0; // counts run throughs

    // counter to open file
    // 0 to 99 are reserved for train data
    int qpg_iter = 100;
    int nqpg_iter = 100;

    for(int i = 0; i < nFiles_test; i++){

        // takes randomly a nqgp or qgp file
        if(rand() > (RAND_MAX / 2)){

            // saves a nqgp file
            fileAddress = QString::fromStdString("/phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." + to_string(nqpg_iter) + "_event.dat");
            fileNames = nqgp.path() + fileAddress;

            nqpg_iter++;

            // runs data through network
            network->import_data(fileNames.toStdString());
            network->change_input();
            network->calc_data();

            result.push_back(network->network[network->network.size()-1][0].getOutput());
            result.push_back(network->network[network->network.size()-1][1].getOutput());
            current_average += result[1];
            result.clear();

        }
        else {
            // saves a qgp file
            fileAddress = QString::fromStdString("/phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." + to_string(qpg_iter) + "_event.dat");
            fileNames = qgp.path() + fileAddress;

            qpg_iter++;

            // runs file through network
            network->import_data(fileNames.toStdString());
            network->change_input();
            network->calc_data();

            result.push_back(network->network[network->network.size()-1][0].getOutput());
            result.push_back(network->network[network->network.size()-1][1].getOutput());
            current_average += result[0];
            result.clear();
        }
        // checks if it was last file and pass it to network
        if(i==nFiles-1){
            network->last_sample = true;
        }
        cnt++;

        // prints out output in console
        /*cout << endl << "Test: " << cnt <<  endl;
        cout<< "qgp: "<<network->network[network->network.size()-1][0].getOutput() << endl;
        cout<< "nqgp: " << network->network[network->network.size()-1][1].getOutput() << endl;*/
    }

    //graphic for the average
    current_average /= nFiles_test;
    // pass data into graphs
    double size = x0.size();
    x0.push_back(size);
    y0.push_back(current_average * 100);
    current_average = 0;
    ui->widget->graph(0)->setData(x0, y0);
    ui->widget->replot();
}

void MainWindow::run_input_analysis(){
    QString fileAddress;
    QString fileNames;
    vector<double> averages;
    averages.resize(3);
    ui->widget->graph(0)->setBrush(QBrush(Qt::NoBrush));
    ui->widget->yAxis->setRange(15,17);
    ui->widget->xAxis->setRange(0,nFiles);
    ui->widget->addGraph();
    ui->widget->graph(1)->setPen(QPen(Qt::red)); // line color red for phi angle
    ui->widget->addGraph();
    ui->widget->graph(2)->setPen(QPen(Qt::green)); // line color green for theta angle
    //ui->widget->graph(0)->setBrush(QBrush(QColor(0, 0, 255, 20))); // first graph will be filled with translucent blue
    for(int i = 0; i < nFiles; i++){
        averages = network->analyze_input();
        fileAddress = QString::fromStdString("/phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr." + to_string(i) + "_event.dat");
        if(rand() > (RAND_MAX / 2)){
            fileNames = nqgp.path() + fileAddress;
        }
        else{
            fileNames = qgp.path() + fileAddress;
        }
        network->import_data(fileNames.toStdString());
        network->change_input();
        x2.push_back(x2.size());
        x3.push_back(x2.size());
        x4.push_back(x2.size());
        y2.push_back(averages[0]);      //momentum
        y3.push_back(averages[0]);      //phi angle
        y4.push_back(averages[0]);      //theta angle
        ui->widget->graph(0)->setData(x2, y2);
        ui->widget->graph(1)->setData(x3, y3);
        ui->widget->graph(2)->setData(x4, y4);
        ui->widget->replot();
        QApplication::processEvents();
    }
}

void MainWindow::makePlot()
{
    ui->widget->addGraph();
    ui->widget->graph(0)->setPen(QPen(Qt::blue)); // line color blue for first graph
    ui->widget->graph(0)->setBrush(QBrush(QColor(0, 0, 255, 20))); // first graph will be filled with translucent blue
    ui->widget->yAxis->setRange(0,100);
    ui->widget_2->addGraph();
    ui->widget_2->graph(0)->setPen(QPen(Qt::red)); // line color red for second graph

    //widget graphic average
    // configure right and top axis to show ticks but no labels:
    // (see QCPAxisRect::setupFullAxesBox for a quicker method to do this)
    ui->widget->xAxis2->setVisible(true);
    ui->widget->xAxis2->setTickLabels(false);
    ui->widget->yAxis2->setVisible(true);
    ui->widget->yAxis2->setTickLabels(false);

    // make left and bottom axes always transfer their ranges to right and top axes:
    connect(ui->widget->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->widget->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->widget->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->widget->yAxis2, SLOT(setRange(QCPRange)));
    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->widget->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);

    //widget graphic loss function
    // configure right and top axis to show ticks but no labels:
    // (see QCPAxisRect::setupFullAxesBox for a quicker method to do this)
    ui->widget_2->xAxis2->setVisible(true);
    ui->widget_2->xAxis2->setTickLabels(false);
    ui->widget_2->yAxis2->setVisible(true);
    ui->widget_2->yAxis2->setTickLabels(false);
    ui->widget_2->yAxis2->setRange(0,1);
    ui->widget_2->yAxis->setRange(0,1);
    // make left and bottom axes always transfer their ranges to right and top axes:
    connect(ui->widget_2->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->widget_2->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->widget_2->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->widget_2->yAxis2, SLOT(setRange(QCPRange)));
    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->widget_2->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
}
