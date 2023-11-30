#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include "amr-cpu.h"

using namespace std;

Cell grid[N_cell_max];
unordered_map<idx4, Cell> hashtable;

uint transposeToHilbert(const uint X[N_dim], const int L) {
    uint hindex = 0, n = 0;
    for (short i = 0; i < N_dim; ++i) {
        for (int b = 0; b < L; ++b) {
            n = (b * N_dim) + i;
            hindex |= (((X[N_dim-i-1] >> b) & 1) << n);
        }
    }
    return hindex;
}

void hilbertToTranspose(uint X[N_dim], const uint hindex, const int L) {
    uint h = hindex;
    for (short i = 0; i < N_dim; ++i) X[i] = 0;
    for (short i = 0; i < N_dim * L; ++i) {
        short a = (N_dim - (i % N_dim) - 1);
        X[a] |= (h & 1) << (i / N_dim);
        h >>= 1;
    }
}

/* 
Compute the Hilbert index for a given 4-idx (i, j, k, L)

Args
idx4: 4-index
*/
void getHindex(idx4 idx_cell, uint& hindex) { // Axes to transpose
    uint X[3] = {idx_cell.i, idx_cell.j, idx_cell.k};
    int L = idx_cell.L;
    uint m = 1 << (L - 1), p, q, t;
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
    hindex = transposeToHilbert(X, L);
    // X[0], X[1], X[2] needs to be converted into single number
}

/* 
Compute the 3-index for a given Hilbert index and AMR level

Args
X: Hilbert index transpose
L: AMR level
N_dim: number of dimensions
*/
void getHindexInv(uint hindex, int L, idx4& cell_idx) { // Transpose to axes
    // TODO: convert hindex to X
    uint X[N_dim];
    hilbertToTranspose(X, hindex, L);

    
    uint n = 2 << (L - 1), p, q, t;
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
    cell_idx.i = X[0], cell_idx.j = X[1], cell_idx.k = X[2];
    cell_idx.L = L;
}

void makeBaseGrid(Cell (&grid)[N_cell_max]) {
    int offset;
    for (int L=0; L <= L_base; L++) {
        offset = (pow(2, N_dim * L) - 1) / (pow(2, N_dim) - 1);
        for (int hindex=0; hindex < pow(2, N_dim * L); hindex++) {
            idx4 idx_cell;
            getHindexInv(hindex, L, idx_cell);
            Cell cell;
            double dx = 1 / pow(2, L);
            double x = idx_cell.i * dx + dx / 2;
            double y = idx_cell.j * dx + dx / 2;
            double z = idx_cell.k * dx + dx / 2;
            cell.rho = rhoFunc(x, y, z);
            // TODO: Add flag_leaf and flag_faces
            grid[offset + hindex] = cell;
            hashtable[idx_cell] = grid[offset + hindex];
        }
    }
}

// gaussian
double rhoFunc(const double x, const double y, const double z, const double sigma = 1.0) {
    double point[3] = {x, y, z};
    double rsq = 0;
    for (int i = 0; i < N_dim; i++) {
        rsq += pow(point[i] - 0.5, 2);
    }
    double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
    return rho;
}

bool refCrit(double rho) {
    return rho > rho_crit;
}

void grad_cell(idx4 cell_idx, Cell (&grid)[N_cell_max]) {
    uint i = cell_idx.i;
    uint j = cell_idx.j;
    uint k = cell_idx.k;
    uint L = cell_idx.i;
    Cell cell = hashtable[cell_idx];
}

int main() {
    idx4 idx_cell;
    cin >> idx_cell.i;
    cin >> idx_cell.j;
    cin >> idx_cell.k;
    // idx_cell.i = 0;
    // idx_cell.j = 0;
    // idx_cell.k = 1;
    idx_cell.L = 2;
    uint hindex;
    getHindex(idx_cell, hindex);
    cout << hindex << endl;
}