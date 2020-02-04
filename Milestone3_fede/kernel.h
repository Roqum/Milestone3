#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <random>

using namespace std;
class Kernel
{
   vector<vector<vector<double>>> kernel;
   size_t channels;
public:
    Kernel(size_t channels);

    //get weight operator
    double operator() (size_t a, size_t b, size_t c){
        return kernel[a][b][c];
    }

    //set weight operator
    /*
     *
     */
    void random_weights();
    void reset_weights();
private:
};

#endif // KERNEL_H
