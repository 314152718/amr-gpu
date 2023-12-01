#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include "amr-cpu.h"

using namespace std;

typedef unsigned int uint;

Cell grid[N_cell_max];

// define a hash function for the idx4 struct
// based on the hash function from mini-ramses
template<>
struct hash<idx4> {
    size_t operator()(const idx4& idx_cell) const noexcept {
        int complete_hash = 0;
        for (short i = 0; i < N_dim; i++) {
            complete_hash += idx_cell.idx3[i] * hash_constants[i];
        }
        complete_hash += idx_cell.L * hash_constants[3];
        return complete_hash;
    }
};

// create the hashtable
unordered_map<idx4, Cell> hashtable;
unordered_map<idx4, Cell>:: iterator hashtable_itr;

// convert from transposed Hilbert index to Hilbert index
void transposeToHilbert(const uint X[N_dim], const int L, uint &hindex) {
    uint n = 0;
    hindex = 0;
    for (short i = 0; i < N_dim; ++i) {
        for (int b = 0; b < L; ++b) {
            n = (b * N_dim) + i;
            hindex |= (((X[N_dim-i-1] >> b) & 1) << n);
        }
    }
}

// convert from Hilbert index to transposed Hilbert index
void hilbertToTranspose(const uint hindex, const int L, uint (&X)[N_dim]) {
    uint h = hindex;
    for (short i = 0; i < N_dim; ++i) X[i] = 0;
    for (short i = 0; i < N_dim * L; ++i) {
        short a = (N_dim - (i % N_dim) - 1);
        X[a] |= (h & 1) << (i / N_dim);
        h >>= 1;
    }
}

// Compute the Hilbert index for a given 4-idx (i, j, k, L)
void getHindex(idx4 idx_cell, uint& hindex) {
    uint X[3];
    for (short i = 0; i < N_dim; i++) {
        X[i] = idx_cell.idx3[i];
    }
    int L = idx_cell.L;
    uint m = 1 << (L - 1), p, q, t;
    int i;
    // Inverse undo
    for (q = m; q > 1; q >>= 1) {
        p = q - 1;
        for(short i = X[0]; i < N_dim; i++) {
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
    for (short i = 1; i < N_dim; i++) {
        X[i] ^= X[i-1];
    }
    t = 0;
    for (q = m; q > 1; q >>= 1) {
        if (X[N_dim - 1] & q) {
            t ^= q - 1;
        }
    }
    for (short i = 0; i < N_dim; i++) {
        X[i] ^= t;
    }
    transposeToHilbert(X, L, hindex);
}

// Compute the 3-index for a given Hilbert index and AMR level
void getHindexInv(uint hindex, int L, idx4& idx_cell) {
    uint X[N_dim];
    hilbertToTranspose(hindex, L, X);
    uint n = 2 << (L - 1), p, q, t;
    // Gray decode by H ^ (H/2)
    t = X[N_dim - 1] >> 1;
    for (short i = N_dim - 1; i > 0; i--) {
        X[i] ^= X[i - 1];
    }
    X[0] ^= t;
    // Undo excess work
    for (q = 2; q != n; q <<= 1) {
        p = q - 1;
    }
    for (short i = N_dim - 1; i > 0; i--) {
        if(X[i] & q) { // invert
            X[0] ^= p;
        } else {
            t = (X[0]^X[i]) & p;
            X[0] ^= t;
            X[i] ^= t;
        }
    } // exchange
    for (short i = 0; i < N_dim; i++) {
        idx_cell.idx3[i] = X[i];
    }
    idx_cell.L = L;
}

// Multi-variate Gaussian distribution
double rhoFunc(const double coord[N_dim], const double sigma) {
    double rsq = 0;
    for (short i = 0; i < N_dim; i++) {
        rsq += pow(coord[i] - 0.5, 2);
    }
    double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
    return rho;
}

// Criterion for refinement
bool refCrit(double rho) {
    return rho > rho_crit;
}

// Compute the index of the parent cell
void getParentIdx(const idx4 idx_cell, idx4 idx_parent) {
    for (short i = 0; i < N_dim; i++) {
        idx_parent.idx3[i] = idx_cell.idx3[i] / 2;
    }
    idx_parent.L = idx_cell.L - 1;
}

// Compute the indices of the neighbor cells in a given direction
void getNeighborIdx(const idx4 idx_cell, const uint dir, const bool pos, idx4 &idx_neighbor) {
    for (short i = 0; i < N_dim; i++) {
        idx_neighbor.idx3[i] = idx_cell.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_cell.L;
}

// Check if a cell exists
bool checkIfExists(const idx4 idx_cell) {
    return !(hashtable.find(idx_cell) == hashtable.end());
}

// Check if a cell face in a give direction is a border of the computational domain
void checkIfBorder(const idx4 idx_cell, const uint dir, const bool pos, bool &is_border) {
    is_border = idx_cell.idx3[dir] == int(pos) * (pow(2, idx_cell.L) - 1);
}

void makeBaseGrid(Cell (&grid)[N_cell_max]) {
    idx4 idx_cell;
    for (int L=0; L <= L_base; L++) {
        for (uint hindex = 0; hindex < pow(2, N_dim * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            setGridCell(idx_cell, hindex);
        }
    }
}

void setGridCell(const idx4 idx_cell, const uint hindex) {
    // if (checkIfExists(idx_cell)) throw runtime_error("setting existing cell");

    uint offset;
    double dx, coord[3];
    Cell cell;

    offset = (pow(2, N_dim * idx_cell.L) - 1) / (pow(2, N_dim) - 1);
    dx = 1 / pow(2, idx_cell.L);
    for (short i = 0; i < N_dim; i++) {
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
    }
    cell.rho = rhoFunc(coord);
    cell.flag_leaf = idx_cell.L == L_base; // this is only good for base grid set
    grid[offset + hindex] = cell;
    hashtable[idx_cell] = grid[offset + hindex];
}

void refineGridCell(const idx4 idx_cell) {
    uint hindex;
    getHindex(idx_cell, hindex);

    if (!hashtable[idx_cell].flag_leaf) throw runtime_error("trying to refine non-leaf");
    hashtable[idx_cell].flag_leaf = false;

    idx4 idx_child;
    idx_child.L = idx_cell.L + 1;

    for (short dir = 0; dir < N_dim; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            idx_child.idx3[dir] = idx_cell.idx3[dir]*2 + pos;
            idx4 idx_child_new = idx_child; // how to copy struct?
            setGridCell(idx_child_new, hindex);
            hashtable[idx_cell].flag_leaf = true;
        }
    }
    // refine neighbors if needed
    idx4 idx_neighbor;
    uint hindex_neighbor;
    for (short dir = 0; dir < N_dim; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            bool is_border;
            checkIfBorder(idx_cell, dir, pos, is_border);
            if (is_border) continue;
            getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
            // don't need to remove 'if' statements because this is part not for GPU (only gradient is)
            if (checkIfExists(idx_neighbor)) continue;

            // if not exists, drop L by 1
            idx_neighbor.L -= 1;
            for (short i = 0; i < N_dim; i++) {
                idx_neighbor.idx3[i] /= 2;
            }
            refineGridCell(idx_neighbor);
        }
    }
}

// get information about the neighbor cell necessary for computing the gradient
void getNeighborInfo(const idx4 idx_cell, const uint dir, const bool pos, bool &is_ref, double &rho) {
    idx4 idx_neighbor;
    uint idx1_parent_neighbor;
    bool is_border, is_notref;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    is_notref = checkIfExists(idx_neighbor) && !is_border;
    is_ref = !is_notref && !is_border;
    // if the cell is a border cell, set the neighbor index to the cell index (we just want a valid key for the hashtable)
    // if the neighbor is not refined, set the neighbor index to the index of the parent cell's neighbor
    // if the neighbor is refined, don't change the neighbor index
    for (short i = 0; i < N_dim; i++) {
        idx1_parent_neighbor = idx_cell.idx3[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor.idx3[i] = idx_cell.idx3[i] * int(is_border) + idx_neighbor.idx3[i] * int(is_ref) + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor.L = idx_cell.L - int(is_notref);
    // if the cell is a border cell, use the boundary condition
    rho = hashtable[idx_neighbor].rho * int(!is_border) + rho_boundary * int(is_border);
}

// compute the gradient for one cell
void calcGradCell(idx4 idx_cell) {
    bool is_ref[2];
    double dx, rho[3];
    int fd_case;
    Cell cell = hashtable[idx_cell];
    dx = pow(0.5, idx_cell.L);
    rho[2] = cell.rho;
    for (short dir = 0; dir < N_dim; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_cell, dir, pos, is_ref[pos], rho[pos]);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell.rho_grad[dir] = (fd_kernel[fd_case][0] * rho[0] + fd_kernel[fd_case][1] * rho[2] + fd_kernel[fd_case][2] * rho[1]) / (fd_kernel[fd_case][3] * dx);
    }
    hashtable[idx_cell] = cell;
}

// compute the gradient
void calcGrad() {
    idx4 idx_cell;
    for (hashtable_itr = hashtable.begin(); hashtable_itr != hashtable.end(); hashtable_itr++) {
        idx_cell = hashtable_itr->first;
        calcGradCell(idx_cell);
    }
}

void writeGrid() {
    ofstream outfile;
    outfile.open(outfile_name);
    outfile << "Writing this to a file.\n";
    outfile.close();
}

void test1() {
    idx4 idx_cell;
    cin >> idx_cell.idx3[0];
    cin >> idx_cell.idx3[1];
    cin >> idx_cell.idx3[2];
    idx_cell.L = 2;
    uint hindex;
    getHindex(idx_cell, hindex);
    cout << hindex << endl;
}

int main() {
    makeBaseGrid(grid);
}