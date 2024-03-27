// defines
#define _USE_MATH_DEFINES

// cpu includes
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include "amr-gpu.h"

// gpu includes
#include "cuco/static_map.cuh"
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/atomic>

// namespaces
using namespace std;
using namespace std::chrono;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

typedef matrix_type int***** //thrust::double_access_matrix<double>;

// compute the Hilbert index for a given 4-idx (i, j, k, L)
void getHindex(int idx_level[NDIM+1], int& hindex) {
    int X[NDIM];
    for (int i=0; i<NDIM; i++){
        X[i] = idx_level[i];
    }
    short int L = idx_level[NDIM];
    uint16 m = 1 << (L - 1), p, q, t;
    // Inverse undo
    for (q = m; q > 1; q >>= 1) {
        p = q - 1;
        for(short i = X[0]; i < NDIM; i++) {
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
    for (short i = 1; i < NDIM; i++) {
        X[i] ^= X[i-1];
    }
    t = 0;
    for (q = m; q > 1; q >>= 1) {
        if (X[NDIM - 1] & q) {
            t ^= q - 1;
        }
    }
    for (short i = 0; i < NDIM; i++) {
        X[i] ^= t;
    }
    transposeToHilbert(X, L, hindex);
}

// compute the 3-index for a given Hilbert index and AMR level
void getHindexInv(int hindex, int L, int *idx_level) {
    int X[NDIM];
    hilbertToTranspose(hindex, L, X);
    int n = 2 << (L - 1), p, q, t;
    // Gray decode by H ^ (H/2)
    t = X[NDIM - 1] >> 1;
    for (short i = NDIM - 1; i > 0; i--) {
        X[i] ^= X[i - 1];
    }
    X[0] ^= t;
    // Undo excess work
    for (q = 2; q != n; q <<= 1) {
        p = q - 1;
    }
    for (short i = NDIM - 1; i > 0; i--) {
        if(X[i] & q) { // invert
            X[0] ^= p;
        } else {
            t = (X[0]^X[i]) & p;
            X[0] ^= t;
            X[i] ^= t;
        }
    } // exchange
    for (int i = 0; i < NDIM; i++) {
        idx_level[i] = X[i];
    }
    idx_level[NDIM] = L;
}

double rhoFunc(const double coord[NDIM], const double sigma) {
    double rsq = 0;
    for (short i = 0; i < NDIM; i++) {
        rsq += pow(coord[i] - 0.5, 2);
    }
    double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
    return rho;
}

// criterion for refinement
bool refCrit(double rho) {
    return rho > rho_crit;
}

// compute the index of the parent cell
void getParentIdx(const int *idx_level, int *idx_level_parent) {
    for (short i = 0; i < NDIM; i++) {
        idx_level_parent[i] = idx_level[i] / 2;
    }
    idx_level_parent[NDIM+1] = idx_level[NDIM+1] - 1;
}

// compute the indices of the neighbor cells on a given face
__host__ __device__ void getNeighborIdx(const int *idx_cell, const int dir, const bool pos, int *idx_neighbor) {
    // after this getNeighborIdx is applied, must check if neighbor exists (border) !!!
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor.idx3[i] = idx_cell.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_cell.L;
}

// check if a given face is a border of the computational domain
__host__ __device__ void checkIfBorder(const int *idx_cell, const int dir, const bool pos, bool &is_border) {
    is_border = idx_cell.idx3[dir] == int(pos) * (pow(2, idx_cell.L) - 1);
}

bool keyExists(const int *idx_cell, matrix_type matrix_grid) {
    return host_table.find(idx_cell) != host_table.end();
}
// cannot return a value on the __global__ kernel, but can on __device__
template <typename Map>
__device__ void keyExists(const int *idx_cell, Map hashtable_ref, bool &res) {
    res = hashtable_ref.find(idx_cell) != hashtable_ref.end();
}

// get information about the neighbor cell necessary for computing the gradient
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
template <typename Map>
__device__ void getNeighborInfo(const int *idx_cell, const int dir, const bool pos, 
                                bool &is_ref, double &rho_neighbor, Map hashtable_ref) {
    int *idx_neighbor = new int[NDIM+1];
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    keyExists(idx_neighbor, hashtable_ref, exists); 
    is_notref = !exists && !is_border;
    is_ref = !is_notref && !is_border;
    // if the cell is a border cell, set the neighbor index to the cell index (we just want a valid key for the hashtable)
    // if the neighbor is not refined, set the neighbor index to the index of the parent cell's neighbor
    // if the neighbor is refined, don't change the neighbor index
    for (short i = 0; i < NDIM; i++) {
        idx1_parent_neighbor = idx_cell.idx3[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor.idx3[i] = idx_cell.idx3[i] * int(is_border) + idx_neighbor.idx3[i] * int(is_ref) 
                                + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor.L = idx_cell.L - int(is_notref);
    // if the cell is a border cell, use the boundary condition
    Cell* pCell = hashtable_ref.find(idx_neighbor)->second;
    rho_neighbor = pCell->rho * int(!is_border) + rho_boundary * int(is_border);
}

// compute the gradient for one cell
template <typename Map>
__device__ void calcGradCell(const int *idx_cell, Cell* cell, Map hashtable_ref) {
    bool is_ref[2];
    double dx, rho[3];
    int fd_case;

    dx = pow(0.5, idx_cell.L);
    rho[2] = cell->rho;

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_cell, dir, pos, is_ref[pos], rho[pos], hashtable_ref);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell->rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                             + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
    }
}

// compute the gradient
template <typename Map, typename KeyIter>
__global__ void calcGrad(Map hashtable_ref, KeyIter contained_keys, size_t num_keys) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    contained_keys += tid;

    while (tid < num_keys) {
        int *idx_cell = *contained_keys;
        Cell *pCell = hashtable_ref.find(idx_cell)->second;
        calcGradCell(idx_cell, pCell, hashtable_ref);

        contained_keys += gridDim.x * blockDim.x;
        tid += gridDim.x * blockDim.x;
    }
}

template <typename KeyIter, typename ValueIter>
void writeGrid(KeyIter keys_iter, ValueIter underl_values_iter, size_t num_keys, string filename) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    for (int i = 0; i < num_keys; i++) {
        int *idx_cell = *keys_iter;
        Cell cell = *underl_values_iter;

        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << cell.flag_leaf << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";

        keys_iter++;
        underl_values_iter++;
    }
    outfile.close();
}

void writeGrid(matrix_type matrix_grid, string filename) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    for (auto kv : host_table) {
        int *idx_cell = kv.first;
        Cell cell = kv.second;
        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << cell.flag_leaf << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
    }
    outfile.close();
}

// initialize the base level grid
void makeBaseGrid(matrix_type matrix_grid) {
    int *idx_level = new int[NDIM+1];
    for (int L = 0; L <= LBASE; L++) {
        for (int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            setGridCell(matrix_grid, idx_level); // cells have flag_leaf == 1 at L == LBASE == 3
        }
    }
};

void print_cell(int *cell) {
    string s = "";
    for (int i = 0; i < NDIM; i++) {
        s += "["+to_string(cell[i])+", ";
    }
    s += to_string(cell[NDIM])+")";
    printf(s);
}

// set a grid cell in the grid array and the hash table
void setGridCell(matrix_type matrix_grid, 
                 const int *idx_level, int32_t flag_leaf) {
    if (keyExists(idx_level, host_table)) throw runtime_error("setting existing cell");

    int offset;
    double dx, coord[3];
    offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);
    dx = 1.0 / pow(2, idx_level[NDIM]);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
        if (idx_level[i] > IDX_MAX) throw runtime_error("idx_level[i] >= IDX_MAX for i="+to_string(i));
    }

    matrix_grid[idx_level[0]][idx_level[1]][idx_level[2]][idx_level[3]][0] = rhoFunc(coord, sigma);
    for (int i = 0; i < NDIM; i++) {
        matrix_grid[idx_level[0]][idx_level[1]][idx_level[2]][idx_level[3]][1+i] = 0.0;
    }
    printf("HOST ");
    print_cell(matrix_grid[idx_level[0]][idx_level[1]][idx_level[2]][idx_level[3]]);
    printf("\n");
}

template <typename KeyIter>
__global__ void printHashtable(KeyIter key_iter, size_t num_keys) { // ??
    printf("GPU hashmap num_keys %d\n", num_keys);

    for (int i = 0; i < num_keys; i++) {
        int idx_level[NDIM+1] = *zipped;
        Cell* pCell = hashtable_ref.find(idx_cell)->second; //t.get<1>();
        
        printf("GPU ");
        idx_cell.print();
        if (!pCell) printf("ERROR: accessing null cell ptr");
        pCell->print();
        printf("\n");
        zipped++;
    }
}

void test_gradients_baseGrid() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3
    matrix_type matrix_grid = new matrix_type[IDX_MAX+1][IDX_MAX+1][IDX_MAX+1][LMAX+1][NDIM+1]; // ??
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(matrix_grid);
    writeGrid(matrix_grid, "grid-host.csv");

    cout << "Calculating gradients" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU

    calcGrad<<<GRID_SIZE, BLOCK_SIZE>>>(matrix_grid);

    cudaDeviceSynchronize();
    // is this reference going to work?
    amr_gpu::CHECK_LAST_CUDA_ERROR();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    
    writeGrid(matrix_grid, "grid-device.csv");
    printHashtable<<<1, 1>>>(matrix_grid, matrix_grid.size());
}

int main() {
    try {
        test_gradients_baseGrid();
    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
