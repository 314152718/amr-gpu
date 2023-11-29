#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "amr-cpu.h"

using namespace std;

struct idx4 {
    int i, j, k, L;
};

struct Cell {
    int i, j, k, L;
    double rho;
    int flag_leaf;
    int flag_faces[6];
};

int transposeToHilbert(const unsigned int* X, const int L) {
    unsigned int h = 0, n = 0;
    for (short i = 0; i < N_dim; ++i) {
        for (int b = 0; b < L; ++b) {
            n =  (b * N_dim) + i;
            h |= (((X[N_dim-i-1] >> b) & 1) << n);
        }
    }
    return h;
}

void hilbertToTranspose(unsigned int* X, const unsigned int h, const int L) {
    // TODO, modify X in place
}

/* 
Compute the Hilbert index for a given 4-idx (i, j, k, L)

Args
idx4: 4-index
*/
void hindex(idx4 idx_cell, int& h_index) { // Axes to transpose
    int X[3] = {idx_cell.i, idx_cell.j, idx_cell.k};
    int L = idx_cell.L;
    unsigned int m = 1 << (L - 1), p, q, t;
    int i;
    // Inverse undo
    for (q = m; q > 1; q >>= 1) {
        p = q - 1;
        for( i = X[0]; i < N_dim; i++ ) {
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
    // convert transpose to hindex
    transposeToHilbert(X, L);
    // X[0], X[1], X[2] needs to be converted into single number
    return hindex;
}

/* 
Compute the 3-index for a given Hilbert index and AMR level

Args
X: Hilbert index transpose
L: AMR level
N_dim: number of dimensions
*/
void hindexInv(unsigned int hindex, int L, int N_dim, idx4& cell_idx) { // Transpose to axes
    // TODO: convert hindex to X
    hilbertToTranspose(hindex, L);

    
    unsigned int n = 2 << (L - 1), p, q, t;
    int i;
    // Gray decode by H ^ (H/2)
    t = X[N_dim - 1] >> 1;
    for (i = N_dim - 1; i > 0; i--) {
        X[i] ^= X[i - 1];
    }
    X[0] ^= t;
    // Undo excess work
    for (q = 2; q != n; q <<= 1) {
        p = q - 1;
    }
    for (i = N_dim - 1; i > 0 ; i--) {
        if(X[i] & q) { // invert
            X[0] ^= p;
        } else {
            t = (X[0]^X[i]) & p;
            X[0] ^= t;
            X[i] ^= t;
        }
    } // exchange
    idx4 cell_idx;
    cell_idx.i = X[0], cell_idx.j = X[1], cell_idx.k = X[2];
    cell_idx.L = L;
    return cell_idx;
}

void make_base_grid() {
    int offset;
    for (int L=0; L <= L_base; L++) {
        offset = (pow(2, N_dim * L) - 1) / (pow(2, N_dim) - 1);
        for (int hindex=0; hindex < pow(2, N_dim * L); hindex++) {
            idx4 idx_cell;
            hindexInv(hindex, L, idx_cell);
            Cell cell;
            double dx = 1 / pow(2, L);
            double x = idx_cell.i * dx + dx / 2;
            double y = idx_cell.j * dx + dx / 2;
            double z = idx_cell.k * dx + dx / 2;
            cell.rho = rhoFunc(x, y, z);
            grid[offset + hindex] = cell;
        }
    }
}

double rhoFunc(double x, double y, double z) {
    double point[3] = {x, y, z};
    double rsq = 0;
    for (int i = 0; i < N_dim; i++) {
        rsq += pow(point[i] - 0.5, 2);
    }
    double rho = exp( -rsq / 2 ) / pow(2 * M_PI, 0.5);
    return rho;
}

int main() {
    unsigned int X[3] = {20, 10, 30};
    hindex(X, 5, N_dim);
    cout << X[0] << endl;
    
    // testing transposeToHilbert
    
    unsigned int X[] = {22, 8, 25};
    cout << transposeToHilbert(X, 5) << endl; // L = 5
}