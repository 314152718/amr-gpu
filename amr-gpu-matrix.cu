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

__device__ float rhoFunc(const double coord[NDIM], const double sigma) {
    double rsq = 0;
    for (short i = 0; i < NDIM; i++) {
        rsq += pow(coord[i] - 0.5, 2);
    }
    float rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
    return rho;
}

// criterion for refinement
bool refCrit(float rho) {
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
    res = 0 <= index && index < num_inserted;
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
                                bool &is_ref, float &rho_neighbor, 
                                DevicePtr gpu_1d_grid_it, uint32_t num_inserted) {
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
    if (is_notref)
        printf("is_notref %d ni %d nj %d nk %d exists %d is_border %d  i %d j %d k %d\n", is_notref, idx_neighbor[0], 
            idx_neighbor[1], idx_neighbor[2], exists, is_border, idx_level[0], idx_level[1], idx_level[2]);

    for (short i = 0; i < NDIM; i++) {
        idx1_parent_neighbor = idx_level[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor[i] = idx_level[i] * int(is_border) + idx_neighbor[i] * int(is_ref) 
                                + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor[NDIM] = idx_level[NDIM] - int(is_notref);
    if (idx_neighbor[NDIM] > LMAX) {
        print_idx_level(idx_neighbor);
        printf("\nERROR: getNeighborInfo L > LMAX; L %d\n", idx_neighbor[NDIM]);
    }
    // if the cell is a border cell, use the boundary condition
    long int neigh_index;
    getIndex(idx_neighbor, neigh_index);
    
    // not supposed to be true
    if (neigh_index >= num_inserted) {
        print_idx_level(idx_neighbor);
        printf("\nERROR: index >= num_inserted; neigh_index %ld num_inserted %u", neigh_index, num_inserted);
    }

    Cell cell = gpu_1d_grid_it[neigh_index];
    rho_neighbor = cell.rho * int(!is_border) + rho_boundary * int(is_border);

    long int index;
    getIndex(idx_level, index);

    /*int val[4] = {1, 1, 1, 2};
    //printf("equals: %d  dir %d pos %d  i %d j %d k %d L %d  index %ld", equals(idx_level, val), idx_level[0], idx_level[1], 
    //    idx_level[2], idx_level[3], index);
    if (equals(idx_level, val)) {
        printf("cell 1 1 1 2; idx_neighbor i %d j %d k %d L %d  dir %d pos %d rho_neighbor %f\n", idx_neighbor[0], idx_neighbor[1],
            idx_neighbor[2], idx_neighbor[3], dir, pos, rho_neighbor);
    }*/
}

// compute the gradient for one cell
template <typename DevicePtr>
__device__ void calcGradCell(int *idx_level, long int index, DevicePtr gpu_1d_grid_it, uint32_t num_inserted) {
    bool is_ref[2];
    double dx;
    float rho[3];
    int fd_case;

    /*int val[4] = {1, 1, 1, 2};
    if (equals(idx_level, val)) {
        printf("cell 1 1 1 2; calcGradCell0 index %ld \n", index);
    }*/
    Cell cell = *(gpu_1d_grid_it + index);
    dx = pow(0.5, idx_level[NDIM]);
    rho[2] = cell.rho;

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_level, dir, pos, is_ref[pos], rho[pos], gpu_1d_grid_it, num_inserted);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell.rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                            + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
        /*if (equals(idx_level, val)) {
            printf("cell 1 1 1 2; calcGradCell dir %d rho0 %.2f rho1 %.2f rho2 %.2f fd_case %d \n", dir, rho[0], rho[1], rho[2], 
                fd_case);
        }*/
    }
    gpu_1d_grid_it[index] = cell;
}

// compute the gradient
// do calc grad for a specific level so that we loop over index
template <typename DevicePtr>
__global__ void calcGrad(DevicePtr gpu_1d_grid_it, int L, uint32_t num_inserted) {
    long int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < num_inserted) {
        long int index = tid;
        int idx_level[NDIM+1];
        getIndexInv(index, L, idx_level);

        for (int i = 0; i < NDIM; i++) {
            if (idx_level[i] >= pow(2, L) || idx_level[i] < 0) {
                print_idx_level(idx_level);
                printf("\nERROR: calcGrad idx_level[%d] out of bounds; idx_level[%d] %d\n", i, i, idx_level[i]);
                printf("calcGrad print start %u %ld %ld %d\n", num_inserted, tid, tid, L);
            }
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
    for (long int index = 0; index < pow(2, NDIM * L); index++) {
        getIndexInv(index, L, idx_level);

        Cell cell = gpu_1d_grid[index];
        
        outfile << idx_level[0] << "," << idx_level[1] << "," << idx_level[2]
                << "," << idx_level[3] << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
    }
    outfile.close();
}

// set a grid cell in the grid array and the hash table
template <typename DevicePtr>
__device__ void setGridCell(DevicePtr gpu_1d_grid_it, const long int index, const int *idx_level, uint32_t num_inserted) {
    double dx, coord[3];
    dx = 1.0 / pow(2, idx_level[NDIM]);

    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_level[i] * dx + dx / 2;
        if (idx_level[i] >= pow(2, idx_level[NDIM])) 
            printf("ERROR idx_level[i] >= pow(2, idx_level[NDIM]) for i=%d\n", i);
    }

    Cell cell = *(gpu_1d_grid_it + index);
    if (!(cell == Cell())) printf("ERROR setting existing cell\n");

    gpu_1d_grid_it[index] = Cell(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, -1);
}

// initialize the base level grid
template <typename DevicePtr>
__global__ void makeBaseGrid(DevicePtr gpu_1d_grid_it, int L, uint32_t num_inserted) {
    long int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < num_inserted) {
        long int index = tid;
        int idx_level[NDIM+1];
        getIndexInv(index, L, idx_level);

        if (idx_level[NDIM] > LMAX) {
            print_idx_level(idx_level);
            printf("\nERROR: makeBaseGrid L > LMAX; L %d\n", idx_level[NDIM]);
        }

        if (index >= num_inserted) printf("ERROR index >= gpu_1d_grid.size()\n");

        setGridCell(gpu_1d_grid_it, index, idx_level, num_inserted); // cells have flag_leaf == 1 at L == LBASE == 3
        
        /*printf("HOST ");
        print_idx_level(idx_level);
        printf(" %ld ", index);
        Cell cell = *gpu_1d_grid_it;
        cell.println();*/
        
        tid += gridDim.x * blockDim.x;
    }
};

template <typename DevicePtr>
__global__ void printHashtable(DevicePtr gpu_1d_grid_it, int L, uint32_t num_inserted) {
    printf("GPU hashmap num_keys %u\n", num_inserted);

    for (long int index = 0; index < pow(2, NDIM * L); index++) {
        int idx_level[NDIM+1];
        getIndexInv(index, L, idx_level);
        Cell cell = gpu_1d_grid_it[index];
        
        printf("GPU ");
        print_idx_level(idx_level);
        printf(" %ld ", index);
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

void test_device_vector_speed() {
    int L = LMAX;
    uint32_t num_inserted = pow(2, NDIM * L);
    if (num_inserted > NCELL_MAX_ARR[L]) 
        throw runtime_error("num_inserted > NCELL_MAX_ARR[L] "+to_string(num_inserted)+to_string(NCELL_MAX_ARR[L]));

    thrust::device_vector<Cell> gpu_1d_grid(num_inserted);
    thrust::device_vector<double> double_arr_grid(num_inserted);

    for (int i = 0; i < 50; i++) {
        auto start = high_resolution_clock::now();
        gpu_1d_grid[num_inserted/2] = Cell();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "Cell set time " << duration.count()/1000.0 << " ms" << endl;


        start = high_resolution_clock::now();
        double_arr_grid[num_inserted/2] = 1.0;
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "double set time " << duration.count()/1000.0 << " ms" << endl << endl;
    }
}

double time_calcGrad(int block_size, int L, thrust::device_vector<Cell> &gpu_1d_grid, int repeat=1) {

    auto grid_size = (NCELL_MAX_ARR[L] + block_size - 1) / block_size;
    printf("num threads: %ld\n", grid_size*block_size);

    double avg_time = 0;
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
        avg_time += duration.count()/1000.0;
        if (repeat > 1) {
            cout << "block_size = " << block_size << ", time: " << duration.count()/1000.0 << " ms" << endl;
        }
    }
    return avg_time / repeat;
}

double time_calcGrad(int block_size, int L, thrust::device_vector<Cell> &gpu_1d_grid, 
    unordered_map<int, double> &times, int repeat=1) {
    if (repeat <= 1 && times.find(block_size) != times.end()) 
        return times.find(block_size)->second;

    double duration = time_calcGrad(block_size, L, gpu_1d_grid, repeat);
    times.insert({block_size, duration});
    return duration;
}

void test_speed_binary() {
    auto start = high_resolution_clock::now();

    int L = LMAX;
    int block_size = 512;
    auto grid_size = (NCELL_MAX_ARR[L] + block_size - 1) / block_size;
    uint32_t num_inserted = pow(2, NDIM * L);
    if (num_inserted > NCELL_MAX_ARR[L]) 
        throw runtime_error("num_inserted > NCELL_MAX_ARR[L] "+to_string(num_inserted)+to_string(NCELL_MAX_ARR[L]));

    // print other block size times
    // and then dont do binary search
    // cudaMalloc array unified memory
    thrust::device_vector<Cell> gpu_1d_grid(num_inserted);
    makeBaseGrid<<<grid_size, block_size>>>(gpu_1d_grid.begin(), 
                                            L,
                                            gpu_1d_grid.size());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    int m = 32, M = 512; // 512 + 256
    unordered_map<int, double> times;
    double min_kernel_time = time_calcGrad(m, L, gpu_1d_grid, times);
    //cout << "STEP2" << endl;
    double max_kernel_time = time_calcGrad(M, L, gpu_1d_grid, times);
    //cout << "STEP3" << endl;
    double min_time = min_kernel_time;
    double max_time = max_kernel_time;

    double mid_time = min_time;
    double mid_up_time, mid_down_time;
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
    time_calcGrad(mid, L, gpu_1d_grid, times, 50);
    cout << endl;

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total time: " << duration.count()/1000.0 << " ms" << endl;
}

void test_speed() {
    auto start_all = high_resolution_clock::now();

    ofstream outfile;
    outfile.open("time_matrix.csv");
    outfile << "L,dur_ms\n";
    int block_size = 512;

    for (int L = LBASE; L <= LMAX; L++) {
        auto start0 = high_resolution_clock::now();
        auto start = start0;

        auto grid_size = (NCELL_MAX_ARR[L] + block_size - 1) / block_size;
        uint32_t num_inserted = pow(2, NDIM * L);
        if (num_inserted > NCELL_MAX_ARR[L]) 
            throw runtime_error("num_inserted > NCELL_MAX_ARR[L] "+to_string(num_inserted)+to_string(NCELL_MAX_ARR[L]));

        // print other block size times
        // and then dont do binary search
        // cudaMalloc array unified memory
        thrust::device_vector<Cell> gpu_1d_grid(num_inserted);   

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after device_vector creation: " << duration.count()/1000.0 << " ms" << endl;

        makeBaseGrid<<<grid_size, block_size>>>(gpu_1d_grid.begin(), 
                                                L,
                                                gpu_1d_grid.size());
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after makeBaseGrid: " << duration.count()/1000.0 << " ms" << endl;

        cout << "L = " << L << ", block size " << block_size << endl;
        double avg_time = time_calcGrad(block_size, L, gpu_1d_grid, 10);

        outfile << L << "," << avg_time << endl;

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start0);
        cout << endl;
        cout << "Total level time for L=" << L << ": " << duration.count()/1000.0 << " ms" << endl << endl;
    }
    outfile.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start_all);
    cout << "Total time: " << duration.count()/1000.0 << " ms" << endl;
}

void test_gradients_baseGrid() {
    // why are some of the non-2 level cells leaves?
    // explicitly use NDIM == 3
    
    uint32_t num_inserted = pow(2, NDIM * LBASE);
    if (num_inserted > NCELL_MAX_ARR[LBASE]) 
        throw runtime_error("num_inserted > NCELL_MAX_ARR[LBASE] "+to_string(num_inserted)+to_string(NCELL_MAX_ARR[LBASE]));
    // time the rest of the code
    // do 64 and 1024 block_size for 1d_vector and hashtable

    thrust::device_vector<Cell> gpu_1d_grid(num_inserted);
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid<<<GRID_SIZE, BLOCK_SIZE>>>(gpu_1d_grid.begin(), 
                                            LBASE,
                                            gpu_1d_grid.size());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
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
        /*if (NCELL_MAX != lround(2*pow(2, LMAX*NDIM))) {
            throw runtime_error("NCELL_MAX != 2*2^(LMAX*NDIM); NCELL_MAX "+to_string(NCELL_MAX)+" 2*2^(LMAX*NDIM) "
                +to_string(lround(2*pow(2, LMAX*NDIM))));
        }*/

        test_speed();

    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
