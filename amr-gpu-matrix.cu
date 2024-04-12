// defines
#define _USE_MATH_DEFINES

// cpu includes
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
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
        for (short i = 0; i < NDIM; i++) {
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

__device__ void getHindexAndOffset(long int hindex_plus_offset, long int &hindex, long int &offset, int &L) {
    // long double is treated as double in device code -> its not
    long double x = hindex_plus_offset;
    x *= (pow(2, NDIM) - 1);
    L = floor(log(x + 1, pow(2, NDIM))); // + EPS); <- only for L > 20
    
    offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
    hindex = hindex_plus_offset - offset;
    if (hindex < 0) {
        printf("\nERROR: hindex < 0; L %d %ld %Lf\n", L, hindex_plus_offset, x + 1);
    }
}


__host__ __device__ void print_idx_level(const int *idx_level, const char *append = "") {
    // explicitly use NDIM == 3
    printf("%s[%d, %d, %d](L=%d)", append, idx_level[0], idx_level[1], idx_level[2], idx_level[3]);
}
__host__ __device__ void print_idx_level_ptr(int *idx_level) {
    // explicitly use NDIM == 3
    printf("[%d, %d, %d](L=%d)", idx_level[0], idx_level[1], idx_level[2], idx_level[3]);
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
        for (short i = NDIM - 1; i >= 0; i--) {
            if (X[i] & q) { // invert
                X[0] ^= p;
            } else {
                t = (X[0]^X[i]) & p;
                X[0] ^= t;
                X[i] ^= t;
            }
        } 
    }
    // exchange
    for (int i = 0; i < NDIM; i++) {
        idx_level[i] = X[i];
    }
    idx_level[NDIM] = L;
}

__host__ __device__ void getIndexInv(long int index, int L, int *idx_level) {
    for (int dim = 0; dim < NDIM; dim++) {
        long int scale = (long int)pow(2, (NDIM-1-dim)*L);
        idx_level[dim] = int(index / scale);
        index %= scale;
    }
    idx_level[NDIM] = L;
}

__host__ __device__ void getIndex(const int *idx_level, long int &index) {
    long int scale = 1;
    index = 0;
    for (int dim = NDIM-1; dim >= 0; dim--) {
        index += scale * idx_level[dim];
        //printf("index %ld scale %ld\n", index, scale);
        scale *= int(pow(2, idx_level[NDIM]));
    }
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
__host__ __device__ void getNeighborIdx(const int *idx_level, const int dir, const bool pos, 
                                        int *idx_neighbor) {
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
    long int index;
    getIndex(idx_level, index);
    long int offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);
    res = 0 <= offset + index && offset + index < num_inserted;
}

__device__ bool equals(const int *idx_level, const int *idx_other) {
    bool res = true;
    for (int i = 0; i < NDIM+1; i++) {
        res = res && idx_level[i] == idx_other[i];
    }
    return res;
}

// get information about the neighbor cell necessary for computing the gradient
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
template <typename DevicePtr>
__device__ void getNeighborInfo(const int *idx_level, const int dir, const bool pos, 
                                bool &is_ref, double &rho_neighbor, 
                                DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    int idx_neighbor[NDIM+1];
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the cell is a border cell
    checkIfBorder(idx_level, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_level, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
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
    if (is_notref)
        printf("is_notref %d\n", is_notref);
    if (idx_neighbor[NDIM] > LMAX) {
        print_idx_level(idx_neighbor);
        printf("\nERROR: getNeighborInfo L > LMAX; L %d\n", idx_neighbor[NDIM]);
    }
    // if the cell is a border cell, use the boundary condition
    long int neigh_offset = (pow(2, NDIM * idx_neighbor[NDIM]) - 1) / (pow(2, NDIM) - 1);
    long int neigh_index;
    getIndex(idx_neighbor, neigh_index);
    // not supposed to be true
    if (neigh_offset + neigh_index >= num_inserted) {
        print_idx_level(idx_neighbor);
        printf("\nERROR: offset + index >= num_inserted; neigh_offset %ld neigh_index %ld num_inserted %lu", 
            neigh_offset, neigh_index, num_inserted);
    }
    if (equals(idx_level, idx_) )

    Cell cell = gpu_1d_grid_it[neigh_offset + neigh_index];
    rho_neighbor = cell.rho * int(!is_border) + rho_boundary * int(is_border);
}

// compute the gradient for one cell
template <typename DevicePtr>
__device__ void calcGradCell(int *idx_level, long int cell_idx, DevicePtr gpu_1d_grid_it, size_t num_inserted) {
    bool is_ref[2];
    double dx, rho[3];
    int fd_case;

    Cell cell = *(gpu_1d_grid_it + cell_idx);
    dx = pow(0.5, idx_level[NDIM]);
    rho[2] = cell.rho;

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_level, dir, pos, is_ref[pos], rho[pos], gpu_1d_grid_it, num_inserted);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell.rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                            + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
    }
    gpu_1d_grid_it[cell_idx] = cell;
}

// compute the gradient
// do calc grad for a specific level so that we loop over index
template <typename DevicePtr>
__global__ void calcGrad(DevicePtr gpu_1d_grid_it, int L, size_t num_inserted) {
    long int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long int offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);

    while (tid < num_inserted) {
        long int index;
        int idx_level[NDIM+1];
        getIndexInv(index, L, idx_level);

        if (idx_level[NDIM] > LMAX) {
            print_idx_level(idx_level);
            printf("\nERROR: calcGrad L > LMAX; L %d\n", idx_level[NDIM]);
            printf("calcGrad print start %lu %ld %ld %ld %d\n", num_inserted, tid, index, offset, L);
        }
        calcGradCell(idx_level, tid, gpu_1d_grid_it, num_inserted);

        tid += gridDim.x * blockDim.x;
    }
}

void writeGrid(thrust::device_vector<Cell> &gpu_1d_grid, string filename, int L) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    int idx_level[NDIM+1];

    outfile << "i,j,k,L,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    long int offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
    for (long int index = 0; index < pow(2, NDIM * L); index++) {
        getIndexInv(index, L, idx_level);

        Cell cell = gpu_1d_grid[offset + index];
        
        outfile << idx_level[0] << "," << idx_level[1] << "," << idx_level[2]
                << "," << idx_level[3] << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
    }
    outfile.close();
}

// set a grid cell in the grid array and the hash table
void setGridCell(thrust::device_vector<Cell> &gpu_1d_grid, const long int index, const int *idx_level) {
    double dx, coord[3];
    long int offset = (pow(2, NDIM * idx_level[NDIM]) - 1) / (pow(2, NDIM) - 1);
    dx = 1.0 / pow(2, idx_level[NDIM]);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_level[i] * dx + dx / 2;
        if (idx_level[i] > IDX_MAX) throw runtime_error("idx_level[i] >= IDX_MAX for i="+to_string(i));
    }

    if (offset + index >= NCELL_MAX) throw runtime_error("offset + index >= N_cell_max");
    if (offset + index >= gpu_1d_grid.size()) throw runtime_error("offset + index >= gpu_1d_grid.size()");
    Cell cell = gpu_1d_grid[offset + index];
    if (!(cell == Cell())) throw runtime_error("setting existing cell");

    gpu_1d_grid[offset + index] = Cell(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, -1);

    /*printf("HOST ");
    print_idx_level(idx_level);
    printf(" %ld %ld ", index, offset);
    cell = gpu_1d_grid[offset + index];
    cell.println();*/
}

// initialize the base level grid
void makeBaseGrid(thrust::device_vector<Cell> &gpu_1d_grid, int L) {
    int idx_level[NDIM+1];
    for (long int index = 0; index < pow(2, NDIM * L); index++) {
        getIndexInv(index, L, idx_level);

        if (idx_level[NDIM] > LMAX) {
            print_idx_level(idx_level);
            printf("\nERROR: makeBaseGrid L > LMAX; L %d\n", idx_level[NDIM]);
        }
        setGridCell(gpu_1d_grid, index, idx_level); // cells have flag_leaf == 1 at L == LBASE == 3
    }
};

template <typename DevicePtr>
__global__ void printHashtable(DevicePtr gpu_1d_grid_it, int L, size_t num_inserted) {
    printf("GPU hashmap num_keys %lu\n", num_inserted);
    long int offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);

    for (long int index = 0; index < pow(2, NDIM * L); index++) {
        int idx_level[NDIM+1];
        getIndexInv(index, L, idx_level);
        Cell cell = gpu_1d_grid_it[offset + index];
        
        printf("GPU ");
        print_idx_level(idx_level);
        printf(" %ld %ld ", index, offset);
        cell.println();
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

void test_upd_arr_ptr(int *ass) {
    ass[0] = 1;
}

void test_arr_ptr() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3

    int *ass = new int[1];
    test_upd_arr_ptr(ass);
    print_idx_level_ptr(ass);
}

void test_index_funcs() {
    for (int L = 0; L <= LBASE; L++) {
        long int offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
        for (int i = 0; i < pow(2, L); i++) {
            for (int j = 0; j < pow(2, L); j++) {
                for (int k = 0; k < pow(2, L); k++) {
                    long int index;
                    int idx_level[4] = {i, j, k, L};
                    int idx_level2[4];
                    getIndex(idx_level, index);
                    getIndexInv(index, L, idx_level2);
                    cout << i << " " << j << " " << k << " " << L << "  " << idx_level[0] << " " << idx_level[1] << " " << 
                        idx_level[2] << " " << idx_level[3] << "  " << idx_level2[0] << " " << idx_level2[1] << " " << 
                        idx_level2[2] << " " << idx_level2[3] << "  " << index << " " << offset << endl;
                }
            }
        }
    }
}

long int time_calcGrad(int block_size, int L, thrust::device_vector<Cell> &gpu_1d_grid, 
                       unordered_map<int, long int> &times, int repeat=1) {
    if (repeat <= 1 && times.find(block_size) != times.end()) 
        return times.find(block_size)->second;

    auto grid_size = (NCELL_MAX + block_size - 1) / block_size;
    printf("num threads: %ld\n", grid_size*block_size);

    auto duration = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    for (int i = 0; i < repeat; i++) {
        auto start = high_resolution_clock::now();

        // run as kernel on GPU

        calcGrad<<<grid_size, block_size>>>(gpu_1d_grid.begin(), 
                                            L,
                                            gpu_1d_grid.size());
        //printf("time_calcGrad calcGrad end\n");

        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

        auto stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        if (repeat > 1) {
            cout << "block_size = " << block_size << ", time: " << duration.count()/1000.0 << " ms" << endl;
        }
    }
    return duration.count()/1000.0;
}

void test_speed() {
    for (int32_t L = LMAX; L <= LMAX; L++) {
        long int num_inserted = 0;
        for (int lvl = 0; lvl <= L; lvl++) {
            for (long int index = 0; index < pow(2, NDIM * lvl); index++) {
                num_inserted++;
            }
        }
        // print other block size times
        // and then dont do binary search
        // cudaMalloc array unified memory
        thrust::device_vector<Cell> gpu_1d_grid(num_inserted);        
        makeBaseGrid(gpu_1d_grid, L);

        int m = 32, M = 512; // 512 + 256
        unordered_map<int, long int> times;
        long int min_kernel_time = time_calcGrad(m, L, gpu_1d_grid, times);
        //cout << "STEP2" << endl;
        long int max_kernel_time = time_calcGrad(M, L, gpu_1d_grid, times);
        //cout << "STEP3" << endl;
        long int min_time = min_kernel_time;
        long int max_time = max_kernel_time;

        long int mid_time = min_time;
        long int mid_up_time, mid_down_time;
        int mid = m;
        
        mid = (M+m)/2/32*32;
        mid_up_time = time_calcGrad(mid+32, L, gpu_1d_grid, times);
        mid_down_time = time_calcGrad(mid-32, L, gpu_1d_grid, times);

        while (M - m > 64 && mid_up_time != mid_down_time) {
            mid = (M+m)/2/32*32;
            mid_time = time_calcGrad(mid, L, gpu_1d_grid, times);
            mid_up_time = time_calcGrad(mid+32, L, gpu_1d_grid, times);
            mid_down_time = time_calcGrad(mid-32, L, gpu_1d_grid, times);
            if (mid_up_time == mid_down_time) {
                if (mid_time > mid_up_time) {
                    printf("ERROR: local max found instead\n");
                }
                break;
            } 
            if (mid_up_time > mid_down_time) {
                M = mid;
                max_time = mid_time;
            } else {
                m = mid;
                min_time = mid_time;
            }
        }
        if (mid_time > min_time) {
            mid = m;
            mid_time = min_time;
        }
        if (mid_time > max_time) {
            mid = M;
            mid_time = max_time;
        }
        cout << "L = " << L << ", best kernel block size: " << mid << " time: " << mid_time << " ms" << endl;
        if (max_kernel_time == min_kernel_time)
            cout << "L = " << L << ", Same time. Min block: " << m << ", max block: " << M << endl;
        //time_calcGrad(mid, L, gpu_1d_grid, times, L, 50);
        cout << endl << endl;
    }
}

void test_gradients_baseGrid() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3
    long int num_inserted = 0;
    for (int L = 0; L <= LBASE; L++) { // only store the last level
        for (long int index = 0; index < pow(2, NDIM * L); index++) {
            num_inserted++;
        }
    }
    // time the rest of the code
    // do 64 and 1024 block_size for 1d_vector and hashtable

    thrust::device_vector<Cell> gpu_1d_grid(num_inserted);
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(gpu_1d_grid, LBASE);
    writeGrid(gpu_1d_grid, "grid-host-matrix.csv", LBASE);

    cout << "num_inserted " << num_inserted << endl;
    //printf("time_calcGrad insert_values\n");


    cout << "\nCalculating gradients\n" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU

    calcGrad<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_1d_grid.begin(),
                                        LBASE,
                                        gpu_1d_grid.size());

    cudaDeviceSynchronize();
    // is this reference going to work?
    CHECK_LAST_CUDA_ERROR();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    
    cout << endl;
    writeGrid(gpu_1d_grid, "grid-device-matrix.csv", LBASE);
    cout << endl;
    printHashtable<<<1, 1>>>(gpu_1d_grid.begin(), LBASE, gpu_1d_grid.size());
}

int main() {
    printf("amr gpu matrix\n");
    try {
        test_gradients_baseGrid();
    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
