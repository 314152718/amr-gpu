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
    uint X[3];
    for (int i = 0; i++; i < N_dim) {
        X[i] = idx_cell.idx3[i];
    }
    int L = idx_cell.L;
    uint m = 1 << (L - 1), p, q, t;
    int i;
    // Inverse undo
    for (q = m; q > 1; q >>= 1) {
        p = q - 1;
        for(i = X[0]; i < N_dim; i++) {
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
void getHindexInv(uint hindex, int L, idx4& idx_cell) { // Transpose to axes
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
    for (i = N_dim - 1; i > 0; i--) {
        if(X[i] & q) { // invert
            X[0] ^= p;
        } else {
            t = (X[0]^X[i]) & p;
            X[0] ^= t;
            X[i] ^= t;
        }
    } // exchange
    for (i = 0; i < N_dim; i++) {
        idx_cell.idx3[i] = X[i];
    }
    idx_cell.L = L;
}

void makeBaseGrid(Cell (&grid)[N_cell_max]) {
    int offset;
    double dx, coord[3];
    idx4 idx_cell;
    Cell cell;
    for (int L=0; L <= L_base; L++) {
        offset = (pow(2, N_dim * L) - 1) / (pow(2, N_dim) - 1);
        dx = 1 / pow(2, L);
        for (int hindex=0; hindex < pow(2, N_dim * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            for (int i = 0; i < N_dim; i++) {
                coord[i] = idx_cell.idx3[i] * dx + dx / 2;
            }
            cell.rho = rhoFunc(coord);
            cell.flag_leaf = L == L_base;
            grid[offset + hindex] = cell;
            hashtable[idx_cell] = grid[offset + hindex];
        }
    }
}

// gaussian
double rhoFunc(const double coord[N_dim], const double sigma = 1.0) {
    double rsq = 0;
    for (int i = 0; i < N_dim; i++) {
        rsq += pow(coord[i] - 0.5, 2);
    }
    double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
    return rho;
}

bool refCrit(double rho) {
    return rho > rho_crit;
}

bool checkIfBorder(const idx4 idx_cell, const uint dir, bool (&isBorder)[2]) {
    isBorder[0] = idx_cell.idx3[dir] == 0;
    isBorder[1] = idx_cell.idx3[dir] == pow(2, idx_cell.L) - 1;
}

void grad_cell(idx4 idx_cell, Cell (&grid)[N_cell_max]) {
    Cell cell = hashtable[idx_cell];
}

int main() {
    idx4 idx_cell;
    cin >> idx_cell.idx3[0];
    cin >> idx_cell.idx3[1];
    cin >> idx_cell.idx3[2];
    // idx_cell.i = 0;
    // idx_cell.j = 0;
    // idx_cell.k = 1;
    idx_cell.L = 2;
    uint hindex;
    getHindex(idx_cell, hindex);
    cout << hindex << endl;
}