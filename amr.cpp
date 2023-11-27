#include <iostream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

const int L_base = 3, L_max = 6, N_dim = 3;

/* 
Compute the Hilbert index

Args
X: spatial index
L: AMR level
N_dim: number of dimensions
*/
void hindex(unsigned int* X, int L, int N_dim ) {
    unsigned int m = 1 << (L - 1), p, q, t;
    int i;
    // Inverse undo
    for (q = m; q > 1; q >>= 1) {
        p = q - 1;
        for( i = 0; i < N_dim; i++ ) {
            if (X[i] & q ) { // invert 
                X[0] ^= p;
            } else { // exchange
                t = (X[0]^X[i]) & p;
                X[0] ^= t;
                X[i] ^= t;
            }
        }
    }
    // Gray encode
    for (i = 1; i < N_dim; i++) {
        X[i] ^= X[i-1];
    }
    t = 0;
    for (q = m; q > 1; q >>= 1) {
        if (X[N_dim - 1] & q) {
            t ^= q - 1;
        }
    }
    for (i = 0; i < N_dim; i++) {
        X[i] ^= t;
    }
}

int main() {
    unsigned int X[3] = {20, 10, 30};
    hindex(X, 5, N_dim);
    cout << X[0] << endl;
}