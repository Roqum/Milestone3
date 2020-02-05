#ifndef KERNEL_H
#define KERNEL_H

#include <vector>
#include <random>

using namespace std;
class Kernel
{
   // Kernel = 3D Matrix of weights
   vector<vector<vector<double>>> kernel;

   size_t channels;
public:
    Kernel(size_t channels);

    ///returns the weight at the given index///
    double operator() (size_t a, size_t b, size_t c){
        return kernel[a][b][c];
    }
    ///initialize random weights///
    void random_weights();

    ///reset all weights to new random weights///
    void reset_weights();
private:
};

#endif // KERNEL_H
