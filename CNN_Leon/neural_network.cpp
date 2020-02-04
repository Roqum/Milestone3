#include "neural_network.h"
#include "bits/stdc++.h"

Neural_Network::Neural_Network(topology net_topology, size_t new_batch_size){
    batch_size = new_batch_size;
    network.resize(net_topology.size());

    // creates the amount of neurons as save in the net_topology
    for(size_t i=0; i<net_topology.size();i++){
        for(size_t j=0; j<net_topology[i];j++){
            // first entry of topology are input Neurons
            if(i==0){
                network[i].push_back(Neuron(nullptr,Neuron::type::Input,net_topology[i+1],j));
            }
            //last entry are number of output layer Neurons
            else if(i==net_topology.size()-1){
                network[i].push_back(Neuron(&network[i-1],Neuron::type::Output,0,j));
            }
            // every other entry are number of hidden layer neurons
            else{
                network[i].push_back(Neuron(&network[i-1],Neuron::type::Hidden,net_topology[i+1],j));
            }

        }
    }
}

// runs and return the sofmax function to the given vector
vector<long double> Neural_Network::softmax_net_output(vector<long double> net_output){
    vector<long double> result;
    result.clear();
    long double sum = 0;
    for(size_t i = 0; i < net_output.size(); i++){
        sum += exp(net_output[i]);
    }
    for(size_t i = 0; i < net_output.size(); i++)
    {
        result.push_back((exp(net_output[i]))/sum);
    }

    return result;
}

void Neural_Network::import_data(string filename){
    raw_input.clear();
    string store_data_line;
    ifstream myFile(filename); // opens file
    if(!myFile.is_open()){
        cout << "Was not able to open the file.\n";
        exit(1);
    }
    // writes the whole file in one string raw_input
    while(myFile.eof() == false){
        getline(myFile, store_data_line);
        raw_input += store_data_line + " ";
    }
    myFile.close();
}


void Neural_Network::change_input(){
    processed_input.clear();
    string number_helper="";

    // adds each number in raw_input to a vector
    for(size_t i=0; i<raw_input.length();i++){
        if(isdigit(raw_input[i])){
            number_helper += raw_input[i];
        }
        else if (number_helper != ""){
            processed_input.push_back(static_cast <double> (stoi(number_helper)));//nur int werte wegen stoi()
            number_helper="";
        }
    }
    raw_input.clear();
}

void Neural_Network::calc_data(){
    // gives each input neuron one number of the file
    for(size_t i=0;i<network[0].size();i++){
        network[0][i].setOutput((processed_input[i]));
        network[0][i].setInput((processed_input[i]));
    }
    processed_input.clear();

    // calculates the values of each Neuron with LReLu function except output Neurons
    for(size_t i=1;i<network.size()-1;i++){
        for(size_t j=0;j<network[i].size();j++){
            network[i][j].calc_Input();
            network[i][j].calc_LReLu_output();
        }
    }

    // calculate the input of all output Neurons
    for(size_t i=0;i<network[network.size()-1].size();i++){
        network[network.size()-1][i].calc_Input();
    }

    // runs softmax with the 2 values of output Neuron
    vector<long double> softmax;
    softmax.clear();
    softmax.push_back(network[network.size()-1][0].getInput());
    softmax.push_back(network[network.size()-1][1].getInput());

    // set each Neuron output its output of the softmax
    network[network.size()-1][0].setOutput(softmax_net_output(softmax)[0]);
    network[network.size()-1][1].setOutput(softmax_net_output(softmax)[1]);
}


void Neural_Network::backpropagation(vector <double> correct_output, int counter){
    size_t index = network.size()-1;
    in_batch_counter +=1;
    batch_done = false;

    // if batch is reached or its the last file
    // then the gradient have to bee updated later on
    if ((in_batch_counter == batch_size) || last_sample){
        batch_done = true;
        last_sample = false;
        in_batch_counter = 0;
    }

    // calculates the gradient beginning from the back of the network
    for(size_t i = 0;i<network.size();i++){
        for(size_t j=0;j<network[index].size();j++){
            if (network[index][j].neuron_type == Neuron::type::Output){
                // calculates gradient of output neurons
                network[index][j].calc_gradient(correct_output[j], batch_size);
            }
            else{
                // calculates gradient if not output neurons
                network[index][j].calc_gradient(batch_size);
            }
            // if batch number is reached, weights are updated
            if(batch_done){
                network[index][j].correct_weights(counter);
                network[index][j].setGradient(0);
            }
        }
        index -= 1;
    }
}

// calculates the loss of network
void Neural_Network::calc_loss_function(vector<double> t, vector<double> o){
    double sum = 0;
    for(int i = 0; i< o.size(); i++){
        sum += pow((t[i]-o[i]),2);
    }
    loss = sum/2;
}

// calculates the average momentum and angles of the particles from the input
// the flattened 4d matrix had the structure of particle_type x momentum x phi-angle x theta-angle
// with that known it is possible to recreate the original bins from the 4d matrix
vector<double> Neural_Network::analyze_input(){
    double momentum = 0;
    double phi = 0;
    double theta = 0;
    size_t particle_counter = 0;
    for (size_t i=0;i<processed_input.size();i++){                      //modulo 8000 is used because the particle type changes every 8000 indexes which is not relevant here
       momentum += processed_input[i] * (20-floor((i%8000)/(20*20)));   //divide by 400 because the momentum changes every 400 indexes       
       phi += processed_input[i] * (floor(((i%8000)%(20*20))/(20)));    //modulo 400 because the phi-angle resets every 400 indexes; divide by 20 because phi-angle changes every 20 indexes
       theta += processed_input[i] * (floor((i%8000)%20));              //theta-angle resets every 20 indexes
       particle_counter += static_cast<size_t>(processed_input[i]);
    }
    momentum = momentum/particle_counter;       //calculates averages by dividing by the total number of particles
    phi = phi/particle_counter;
    theta = theta/particle_counter;
    vector<double> averages{momentum, phi, theta};
    return averages;
}

double Neural_Network::getloss(){
    return loss;
}
