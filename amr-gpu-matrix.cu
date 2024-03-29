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

//typedef matrix_type int***** //thrust::double_access_matrix<double>;

// compute the Hilbert index for a given 4-idx (i, j, k, L)
__host__ __device__ void getHindex(const int *idx_level, long int &hindex) {
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

__device__ double log(double x, double base) {
    return log(x) / log(base);
}

__device__ void getHindexAndOffset(long int hindex_plus_offset, long int &offset, long int &hindex, int &L) {
    L = floor(log(hindex_plus_offset*(pow(2, NDIM) - 1) + 1, pow(2, NDIM)) + EPS);
    offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
    hindex = hindex_plus_offset - offset;
}

// compute the 3-index for a given Hilbert index and AMR level
__host__ __device__ void getHindexInv(long int hindex, int L, int *idx_level) {
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
__host__ __device__ void getNeighborIdx(const int *idx_level, const int dir, const bool pos, int *idx_neighbor) {
    // after this getNeighborIdx is applied, must check if neighbor exists (border) !!!
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor[i] = idx_level[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor[NDIM] = idx_level[NDIM];
}

// check if a given face is a border of the computational domain
__host__ __device__ void checkIfBorder(const int *idx_level, const int dir, const bool pos, bool &is_border) {
    is_border = idx_level[dir] == int(pos) * (pow(2, idx_level[NDIM]) - 1);
}

// cannot return a value on the __global__ kernel, but can on __device__
__device__ void keyExists(const int *idx_level, size_t num_inserted, bool &res) {
    long int hindex;
    getHindex(idx_level, hindex);
    long int offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);
    res = 0 <= offset + hindex && offset + hindex < num_inserted;
}

// get information about the neighbor cell necessary for computing the gradient
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
template <typename DevicePtr>
__device__ void getNeighborInfo(const int *idx_level, const int dir, const bool pos, 
                                bool &is_ref, double &rho_neighbor, 
                                DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    printf("getNeighborInfo start\n");
    int *idx_neighbor = new int[NDIM+1];
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the cell is a border cell
    checkIfBorder(idx_level, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    printf("getNeighborIdx start\n");
    getNeighborIdx(idx_level, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    printf("keyExists start\n");
    keyExists(idx_neighbor, num_inserted, exists); 
    is_notref = !exists && !is_border;
    is_ref = !is_notref && !is_border;
    // if the cell is a border cell, set the neighbor index to the cell index (we just want a valid key for the hashtable)
    // if the neighbor is not refined, set the neighbor index to the index of the parent cell's neighbor
    // if the neighbor is refined, don't change the neighbor index
    for (short i = 0; i < NDIM; i++) {
        idx1_parent_neighbor = idx_level[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor[i] = idx_level[i] * int(is_border) + idx_neighbor[i] * int(is_ref) 
                                + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor[NDIM] = idx_level[NDIM] - int(is_notref);
    // if the cell is a border cell, use the boundary condition
    long int neigh_offset = (pow(2, NDIM * idx_neighbor[NDIM]) - 1) / (pow(2, NDIM) - 1);
    long int neigh_hindex;
    printf("getHindex start\n");
    getHindex(idx_neighbor, neigh_hindex);
    // not supposed to be true
    if (neigh_offset + neigh_hindex >= num_inserted) printf("ERROR: offset + hindex >= num_inserted");

    printf("getNeighborInfo gpu_1d_grid_it start\n");
    Cell cell = gpu_1d_grid_it[neigh_offset + neigh_hindex];
    rho_neighbor = cell.rho * int(!is_border) + rho_boundary * int(is_border);
}

// compute the gradient for one cell
template <typename DevicePtr>
__device__ void calcGradCell(int *idx_level, Cell cell, DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    printf("calcGradCell start\n");
    bool is_ref[2];
    double dx, rho[3];
    int fd_case;

    dx = pow(0.5, idx_level[NDIM]);
    rho[2] = cell.rho;

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            printf("calcGradCell getNeighborInfo start\n");
            getNeighborInfo(idx_level, dir, pos, is_ref[pos], rho[pos], gpu_1d_grid_it, num_inserted);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell.rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                             + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
    }
}

// compute the gradient
template <typename DevicePtr>
__global__ void calcGrad(DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    auto gpu_1d_grid_zipped =
        thrust::make_zip_iterator(thrust::make_tuple(gpu_1d_grid_it));
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    DevicePtr gpu_1d_grid_it2 = DevicePtr(gpu_1d_grid_it);
    gpu_1d_grid_it += tid;

    while (tid < num_inserted) {
        long int offset, hindex;
        int L;
        getHindexAndOffset(tid, offset, hindex, L);
        int *idx_level;
        getHindexInv(hindex, L, idx_level);
        printf("calcGrad print start %d %d %d %d %d\n", num_inserted, tid, threadIdx.x, blockDim.x, blockIdx.x);
        //Cell cell = *gpu_1d_grid_it;
        cell.println();
        cell = *gpu_1d_grid_it;
        cell.println();
        printf("calcGrad calcGradCell start %d %d\n", num_inserted, tid);
        //calcGradCell(idx_level, gpu_1d_grid_it[tid], gpu_1d_grid_it, num_inserted);
        gpu_1d_grid_it += gridDim.x * blockDim.x;
        tid += gridDim.x * blockDim.x;
    }
}

__host__ __device__ void print_idx_level(int *idx_level) {
    string s = "[";
    for (int i = 0; i < NDIM-1; i++) {
        s += to_string(idx_level[i])+", ";
    }
    s += to_string(idx_level[NDIM-1])+"](L="+to_string(idx_level[NDIM])+")\n";
    printf(s.c_str());
}

void writeGrid(thrust::device_vector<Cell> &gpu_1d_grid, string filename) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    int *idx_level = new int[NDIM+1];

    outfile << "i,j,k,L,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    for (int L = 0; L <= LBASE; L++) {
        for (long int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_level);
            long int offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);

            Cell cell = gpu_1d_grid[offset + hindex];
            
            outfile << idx_level[0] << "," << idx_level[1] << "," << idx_level[2]
                    << "," << idx_level[4] << "," << cell.rho << "," << cell.rho_grad[0]
                    << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
        }
    }
    outfile.close();
}

// set a grid cell in the grid array and the hash table
void setGridCell(thrust::device_vector<Cell> &gpu_1d_grid, const long int hindex, const int *idx_level) {
    double dx, coord[3];
    long int offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);
    dx = 1.0 / pow(2, idx_level[NDIM]);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_level[i] * dx + dx / 2;
        if (idx_level[i] > IDX_MAX) throw runtime_error("idx_level[i] >= IDX_MAX for i="+to_string(i));
    }

    if (offset + hindex >= NCELL_MAX) throw runtime_error("offset + hindex >= N_cell_max");
    if (offset + hindex >= gpu_1d_grid.size()) throw runtime_error("offset + hindex >= gpu_1d_grid.size()");
    Cell cell = gpu_1d_grid[offset + hindex];
    if (!(cell == Cell())) throw runtime_error("setting existing cell");

    gpu_1d_grid[offset + hindex] = Cell(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, -1);

    printf("HOST ");
    cell = gpu_1d_grid[offset + hindex];
    cell.println();
}

// initialize the base level grid
void makeBaseGrid(thrust::device_vector<Cell> &gpu_1d_grid) {
    int *idx_level = new int[NDIM+1];
    for (int L = 0; L <= LBASE; L++) {
        for (long int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_level);
            setGridCell(gpu_1d_grid, hindex, idx_level); // cells have flag_leaf == 1 at L == LBASE == 3
        }
    }
};

template <typename DevicePtr>
__global__ void printHashtable(DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    printf("GPU hashmap num_keys %d\n", num_inserted);

    for (int L = 0; L <= LBASE; L++) {
        for (long int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            int *idx_level = new int[NDIM+1];
            getHindexInv(hindex, L, idx_level);
            long int offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
            Cell cell = gpu_1d_grid_it[offset + hindex];
            
            printf("GPU ");
            print_idx_level(idx_level);
            cell.println();
        }
    }
}

void test_device_vector2(thrust::device_vector<Cell> vec) {
    cout << vec[0] << endl;
}

void test_device_vector() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3

    thrust::device_vector<Cell> vec(1);
    vec[0] = Cell(1, 1, 1, 1, 1);
    test_device_vector2(vec);
}

void test_gradients_baseGrid() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3
    long int num_inserted = 0;
    for (int L = 0; L <= LBASE; L++) {
        for (long int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            num_inserted++;
        }
    }

    thrust::device_vector<Cell> gpu_1d_grid(num_inserted);
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(gpu_1d_grid);
    writeGrid(gpu_1d_grid, "grid-host-matrix.csv");

    cout << "num_inserted " << num_inserted << endl;
    //printf("time_calcGrad insert_values\n");


    cout << "Calculating gradients" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU

    calcGrad<<<GRID_SIZE, BLOCK_SIZE>>>(thrust::raw_pointer_cast(gpu_1d_grid.data()), 
                                        gpu_1d_grid.size());

    cudaDeviceSynchronize();
    // is this reference going to work?
    CHECK_LAST_CUDA_ERROR();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    
    writeGrid(gpu_1d_grid, "grid-device-matrix.csv");
    printHashtable<<<1, 1>>>(gpu_1d_grid.begin(), gpu_1d_grid.size());
}

int main() {
    try {
        test_gradients_baseGrid();
    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
