#include <iostream>
#include <vector>
#include <string>

const int N_dim = 3;

using namespace std;

void hilbertToTranspose(unsigned int* X, const unsigned int h, const int L) {
    // TODO, modify X in place
}

int transposeToHilbert(const unsigned int* X, const int L) {
    unsigned int h = 0, n = 0;
    for (short i = 0; i < N_dim; ++i) {
        for (int b = 0; b < L; ++b) {
            n =  (b * N_dim) + i;
            // get nth bit
            // (val >> n) & 1
            h |= (((X[N_dim-i-1] >> b) & 1) << n);
        }
    }
    return h;
}

int main()
{
    unsigned int X[] = {22, 8, 25};
    cout << transposeToHilbert(X, 5) << endl; // L = 5

    X[0] = 0, X[1] = 0, X[2] = 0;

    hilbertToTranspose(X, 22305, 5);

}