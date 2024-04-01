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

// compute the Hilbert index for a given 4-idx (i, j, k, L)
void getHindex(idx4 idx_cell, long int &hindex) {
    int X[NDIM];
    for (int i=0; i<NDIM; i++){
        X[i] = idx_cell.idx3[i];
    }
    short int L = idx_cell.L;
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
void getHindexInv(long int hindex, int L, idx4& idx_cell) {
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
    for (int i=0; i<NDIM; i++) {
        idx_cell.idx3[i] = X[i];
    }
    idx_cell.L = L;
}

// multi-variate Gaussian distribution
/*double rhoFunc(const double coordMid[NDIM], const double cellSide, const double sigma) { // change back
    //int n = round(cellSide / STEP_EPS); // 100 000
    //double eps = cellSide / n;
    //printf("n = %d\n", n);
    int n = 100;
    double eps = cellSide / n;
    double pow_n_NDIM = pow(n, NDIM);

    double coord[NDIM], coordEnd[NDIM], coordStart[NDIM];
    for (int i = 0; i < NDIM; i++) {
        coordStart[i] = coordMid[i] - cellSide/2.0 + eps/2.0; // center of little cell
        coordEnd[i]   = coordMid[i] + cellSide/2.0 - eps/2.0;
        coord[i] = coordStart[i];
    }

    double res = 0.0;
    for (int i = 0; i < pow_n_NDIM; i++) {
        double rsq = 0;
        for (short j = 0; j < NDIM; j++) {
            rsq += pow(coord[j] - 0.5, 2);
        }
        double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
        res += rho;

        if (i == pow_n_NDIM-1) {
            for (int k = 0; k < NDIM; k++) {
                if (!(abs(coord[k] - coordEnd[k]) < STEP_EPS/10.0)) throw runtime_error("did not reach the upper right corner");
            }
            break;
        }

        size_t idx = 0;
        while (idx < NDIM) {
            if (abs(coord[idx] - coordEnd[idx]) < STEP_EPS/10.0) {
                coord[idx] = coordStart[idx];
                idx++;
                if (idx == NDIM) throw runtime_error("impossible: went out of the upper right corner");
            } else break;
        }
        coord[idx] += eps;
    }
    
    return res / pow_n_NDIM;
}*/
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
void getParentIdx(const idx4 &idx_cell, idx4 &idx_parent) {
    for (short i = 0; i < NDIM; i++) {
        idx_parent.idx3[i] = idx_cell.idx3[i] / 2;
    }
    idx_parent.L = idx_cell.L - 1;
}

// compute the indices of the neighbor cells on a given face
__host__ __device__ void getNeighborIdx(const idx4 idx_cell, const int dir, const bool pos, idx4 &idx_neighbor) {
    // after this getNeighborIdx is applied, must check if neighbor exists (border) !!!
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor.idx3[i] = idx_cell.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_cell.L;
}

// check if a given face is a border of the computational domain
__host__ __device__ void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border) {
    is_border = idx_cell.idx3[dir] == int(pos) * (pow(2, idx_cell.L) - 1);
}

bool keyExists(const idx4& idx_cell, host_map &host_table) {
    return host_table.find(idx_cell) != host_table.end();
}
// cannot return a value on the __global__ kernel, but can on __device__
template <typename Map>
__device__ void keyExists(const idx4 idx_cell, Map hashtable_ref, bool &res) {
    res = hashtable_ref.find(idx_cell) != hashtable_ref.end();
}

// get information about the neighbor cell necessary for computing the gradient
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
/*void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_type &hashtable) {
    idx4 idx_neighbor;
    int idx1_parent_neighbor;
    bool is_border, is_notref;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    is_notref = !keyExists(idx_neighbor, hashtable) && !is_border;
    is_ref = !is_notref && !is_border;
    // if the cell is a border cell, set the neighbor index to the cell index (we just want a valid key for the hashtable)
    // if the neighbor is not refined, set the neighbor index to the index of the parent cell's neighbor
    // if the neighbor is refined, don't change the neighbor index
    for (short i = 0; i < NDIM; i++) {
        idx1_parent_neighbor = idx_cell.idx3[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor.idx3[i] = idx_cell.idx3[i] * int(is_border) + idx_neighbor.idx3[i] * int(is_ref) + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor.L = idx_cell.L - int(is_notref);
    // if the cell is a border cell, use the boundary condition
    Cell* pCell = find(hashtable, idx_neighbor);
    rho_neighbor = pCell->rho * int(!is_border) + rho_boundary * int(is_border);
}*/
template <typename Map>
__device__ void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, 
                                bool &is_ref, double &rho_neighbor, Map hashtable_ref) {
    idx4 idx_neighbor;
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
__device__ void calcGradCell(const idx4 idx_cell, Cell* cell, Map hashtable_ref) {
    bool is_ref[2];
    // explicitly use NDIM == 3
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
        idx4 idx_cell = *contained_keys;
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
        idx4 idx_cell = *keys_iter;
        Cell cell = *underl_values_iter;

        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << cell.flag_leaf << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";

        keys_iter++;
        underl_values_iter++;
    }
    outfile.close();
}

void writeGrid(host_map &host_table, string filename) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    for (auto kv : host_table) {
        idx4 idx_cell = kv.first;
        Cell cell = kv.second;
        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << cell.flag_leaf << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
    }
    outfile.close();
}

// initialize the base level grid
void makeBaseGrid(host_map &host_table, int32_t lbase) {
    idx4 idx_cell;
    for (int L = 0; L <= lbase; L++) {
        for (long int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            setGridCell(idx_cell, hindex, L == lbase, host_table); // cells have flag_leaf == 1 at L == lbase == 3
        }
    }
};

// set a grid cell in the grid array and the hash table
void setGridCell(const idx4 idx_cell, const long int hindex, int32_t flag_leaf,
                 host_map &host_table) {
    if (keyExists(idx_cell, host_table)) throw runtime_error("setting existing cell");

    int offset;
    // explicitly use NDIM == 3
    double dx, coord[3];
    offset = (pow(2, NDIM * idx_cell.L) - 1) / (pow(2, NDIM) - 1);
    dx = 1.0 / pow(2, idx_cell.L);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
    }

    // linear 1d index of all cells
    if (offset + hindex >= NCELL_MAX) throw runtime_error("offset () + hindex >= N_cell_max");
    
    host_table[idx_cell] = Cell(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, flag_leaf);
    //printf("HOST ");
    //idx_cell.print();
    //host_table[idx_cell].print();
    //printf("\n");
}

// refine the grid by one level
void refineGrid1lvl(host_map &host_table) {
    for (auto kv : host_table) {
        if (refCrit(kv.second.rho) && kv.second.flag_leaf) {
            refineGridCell(kv.first, host_table);
        }
    }
}

// set child cells in the grid array and hash table
void setGridChildren(idx4 idx_cell, short i, 
                       host_map &host_table) {
    if (i == NDIM) {
        long int hindex;
        getHindex(idx_cell, hindex);
        setGridCell(idx_cell, hindex, 1, host_table);
        return;
    }
    setGridChildren(idx_cell, i+1, host_table);
    idx_cell.idx3[i]++;
    setGridChildren(idx_cell, i+1, host_table);
}

// refine a grid cell
void refineGridCell(const idx4 idx_cell, host_map &host_table) {
    long int hindex;
    getHindex(idx_cell, hindex);
    if (!keyExists(idx_cell, host_table)) throw runtime_error("Trying to refine non-existant cell! "+idx_cell.str());
    Cell cell = host_table[idx_cell];
    if (cell.flag_leaf) throw runtime_error("trying to refine non-leaf");
    if (idx_cell.L == LMAX) throw runtime_error("trying to refine at max level");
    // make this cell a non-leaf
    cell.flag_leaf = 0;
    idx4 idx_child(idx_cell.idx3, size_t(idx_cell.L + 1));
    for (short dir = 0; dir < NDIM; dir++) idx_child.idx3[dir] *= 2;
    // and create 2^NDIM leaf children
    setGridChildren(idx_child, 0, host_table);
    // refine neighbors if needed
    idx4 idx_neighbor, idx_parent;
    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            bool is_border;
            checkIfBorder(idx_cell, dir, pos, is_border);
            if (is_border) continue;
            getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
            if (keyExists(idx_neighbor, host_table)) continue;
            // we assume that L is at most different by 1
            getParentIdx(idx_cell, idx_parent);
            if (!keyExists(idx_parent, host_table))
                throw runtime_error("idx_parent does not exist! "+idx_parent.str()+' '+idx_cell.str());
            getNeighborIdx(idx_parent, dir, pos, idx_neighbor);
            if (!keyExists(idx_neighbor, host_table)) continue; // parent is at border
            refineGridCell(idx_neighbor, host_table);
        }
    }
}

template <typename ValueIter, typename UnderlValueIter>
__global__ void insert_vector_pointers(ValueIter insert_values_begin, 
                                       UnderlValueIter pointer_underl_values_begin, 
                                       size_t num_keys, int* num_inserted) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    pointer_underl_values_begin += tid;

    size_t counter = 0;
    while (tid < num_keys) {
        if (insert_values_begin[tid] = pointer_underl_values_begin) {
            counter++;
        }
        pointer_underl_values_begin += gridDim.x * blockDim.x;
        tid += gridDim.x * blockDim.x;
    }
    atomicAdd(num_inserted, counter);
}

template <typename Map, typename KeyIter, typename ValueIter>
__global__ void insert(Map map_ref,
                       KeyIter key_begin,
                       ValueIter value_begin,
                       size_t num_keys,
                       int* num_inserted) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
 
    size_t counter = 0;
    while (tid < num_keys) {
        // Map::insert returns `true` if it is the first time the given key was
        // inserted and `false` if the key already existed
        thrust::tuple<idx4, Cell*> t = thrust::make_tuple(key_begin[tid], value_begin[tid]);
        idx4 idx_cell = t.get<0>();
        Cell *pCell = t.get<1>();

        // this is ok because all threads go into the if
        if (map_ref.insert(cuco::pair{key_begin[tid], value_begin[tid]})) {
            ++counter;  // Count number of successfully inserted keys
        }

        tid += gridDim.x * blockDim.x;
   }
 
   // Update global count of inserted keys
   atomicAdd(num_inserted, counter);
}

template <typename KeyIter, typename Map>
__global__ void printHashtable(KeyIter key_iter, Map hashtable_ref, int num_keys) {
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(key_iter));
    printf("GPU hashmap num_keys %d\n", num_keys);

    for (int i = 0; i < num_keys; i++) { //auto it = zipped; it != zipped + num_keys; it++
        thrust::tuple<idx4> t = *zipped;
        idx4 idx_cell = t.get<0>();
        Cell* pCell = hashtable_ref.find(idx_cell)->second; //t.get<1>();
        
        printf("GPU ");
        idx_cell.print();
        if (!pCell) printf("ERROR: accessing null cell ptr");
        pCell->print();
        printf("\n");
        zipped++;
    }
}

void test_unordered_map() {
    unordered_map<idx4, Cell> map;
    map[idx4{1, 2, 3, 4}] = Cell{2.0, 4.0, 6.0, 8.0, 0};
    printf("%d\n", map.find(idx4{1, 2, 3, 4}) != map.end());
}

void test_makeBaseGrid() {
    host_map host_table;
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(host_table);
    writeGrid(host_table, "grid-host.csv");
    

    // hashtable insert values from host_table

    thrust::device_vector<idx4> insert_keys(host_table.size());
    thrust::device_vector<Cell> underl_values(host_table.size());
    thrust::device_vector<Cell*> insert_values(host_table.size());

    int i = 0;
    for (auto kv : host_table) {
        insert_keys[i] = kv.first;
        underl_values[i] = kv.second;
        i++;
    }

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      host_table.size(),
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NCELL_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_pcell_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                      insert_keys.begin(),
                                      insert_values.begin(),
                                      host_table.size(),
                                      num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;


    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Cell*> contained_values(num_inserted[0]);
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());

    printHashtable<<<1, 1>>>(contained_keys.begin(), hashtable.ref(cuco::find), num_inserted[0]);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // compare inserted kv
    
    auto tuple_iter =
        thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(),    insert_values.begin(), 
                                                     contained_keys.begin(), contained_values.begin()));
    // Iterate over all slot contents and verify that `slot.key + 1 == slot.value` is always true.

    cout << "Starting thrust::all_of\n";
    auto result = thrust::all_of(
        thrust::device, tuple_iter, tuple_iter + num_inserted[0], [] __device__(auto const& tuple) {
        return thrust::get<0>(tuple) == thrust::get<2>(tuple) && *thrust::get<1>(tuple) == *thrust::get<3>(tuple); // thrust::get<1>(tuple).rho
    });

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    if (result) { cout << "Success! Target values are properly retrieved.\n"; } //incremented
    // keeps failing at comparison but checked in python, GPU is identical to CPU
    else { cout << "Failed at comparison\n"; } //incremented
}

void test_gradients_baseGrid() {
    host_map host_table;
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(host_table);

    /*const int num_ref = LMAX - LBASE;
    cout << "Refining grid levels" << endl;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(host_table);
    }
    cout << "Finished refining grid levels" << endl;*/
    //string filename = "grid-host.csv";
    writeGrid(host_table, "grid-host.csv");
    

    // hashtable insert values from host_table

    thrust::device_vector<idx4> insert_keys(host_table.size());
    thrust::device_vector<Cell> underl_values(host_table.size());
    thrust::device_vector<Cell*> insert_values(host_table.size());

    int i = 0;
    for (auto kv : host_table) {
        insert_keys[i] = kv.first;
        underl_values[i] = kv.second;
        i++;
    }

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      host_table.size(),
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NCELL_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_pcell_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                      insert_keys.begin(),
                                      insert_values.begin(),
                                      host_table.size(),
                                      num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;

    
    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Cell*> contained_values(num_inserted[0]);
    // this is random ordered and is DIFFERENT from insert_keys order
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Calculating gradients" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU

    auto find_ref = hashtable.ref(cuco::find);
    calcGrad<<<GRID_SIZE, BLOCK_SIZE>>>(find_ref, 
                                        contained_keys.begin(), 
                                        num_inserted[0]);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    
    writeGrid(insert_keys.begin(), underl_values.begin(), num_inserted[0], "grid-device.csv");
    printHashtable<<<1, 1>>>(contained_keys.begin(), hashtable.ref(cuco::find), num_inserted[0]);
}


long int time_calcGrad(int block_size, int lbase, host_map &host_table, int repeat=1) {
    
    //printf("time_calcGrad insert_values start\n");
    thrust::device_vector<idx4> insert_keys(host_table.size());
    thrust::device_vector<Cell> underl_values(host_table.size());
    thrust::device_vector<Cell*> insert_values(host_table.size());
    //printf("time_calcGrad insert_values\n");

    int i = 0;
    for (auto kv : host_table) {
        insert_keys[i] = kv.first;
        underl_values[i] = kv.second;
        i++;
    }

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                    thrust::raw_pointer_cast(underl_values.data()),
                                                    host_table.size(),
                                                    num_inserted.data().get());
    //printf("time_calcGrad insert_vector_pointers\n");
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NCELL_MAX>{},
                                    cuco::empty_key{empty_idx4_sentinel},
                                    cuco::empty_value{empty_pcell_sentinel},
                                    thrust::equal_to<idx4>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                    insert_keys.begin(),
                                    insert_values.begin(),
                                    host_table.size(),
                                    num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    
    //printf("time_calcGrad contained_keys start\n");
    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Cell*> contained_values(num_inserted[0]);
    // this is random ordered and is DIFFERENT from insert_keys order
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());
    //printf("time_calcGrad contained_keys end\n");

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    auto find_ref = hashtable.ref(cuco::find);
    auto grid_size = (NCELL_MAX + block_size - 1) / block_size;

    auto duration = duration_cast<microseconds>(high_resolution_clock::now() - high_resolution_clock::now());
    for (int i = 0; i < repeat; i++) {
        auto start = high_resolution_clock::now();

        // run as kernel on GPU

        calcGrad<<<grid_size, block_size>>>(find_ref, 
                                            contained_keys.begin(), 
                                            num_inserted[0]); // add for (50)
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
    for (int32_t lbase = LBASE; lbase <= LMAX; lbase++) {

        //cout << "START" << endl;
        host_map host_table;
        
        makeBaseGrid(host_table, lbase);
        //cout << "STEP1" << endl;

        // hashtable insert values from host_table

        int m = 32, M = 1024;
        long int min_kernel_time = time_calcGrad(m, lbase, host_table);
        cout << "STEP2" << endl;
        long int max_kernel_time = time_calcGrad(M, lbase, host_table);
        cout << "STEP3" << endl;
        long int mid_kernel_time = min_kernel_time;
        int mid = m;
        while (M - m > 32 && max_kernel_time > min_kernel_time) {
            mid = (M+m)/2/32*32;
            mid_kernel_time = time_calcGrad(mid, lbase, host_table);
            if (mid_kernel_time <= min_kernel_time) {
                m = mid;
                min_kernel_time = mid_kernel_time;
            } else {
                M = mid;
                max_kernel_time = mid_kernel_time;
            }
        }
        cout << "lbase = " << lbase << ", best kernel block size: " << mid << endl;
        if (max_kernel_time == min_kernel_time && M - m > 32)
            cout << "lbase = " << lbase << ", Same time. Min block: " << m << ", max block: " << M << endl << endl;
        time_calcGrad(mid, lbase, host_table, 50);
    }
}

void test_GPU_map() {
    //LBASE = 0; // override const

    thrust::device_vector<idx4> insert_keys(1);
    thrust::device_vector<Cell> underl_values(1);
    thrust::device_vector<Cell*> insert_values(1);

    insert_keys[0] = idx4(0, 0, 0, 0);
    underl_values[0] = Cell(1000.00, 0.00, 0.00, 0.00, 1);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      1,
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NCELL_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_pcell_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                      insert_keys.begin(),
                                      insert_values.begin(),
                                      1,
                                      num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;


    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Cell*> contained_values(num_inserted[0]);
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());

    printHashtable<<<1, 1>>>(contained_keys.begin(), hashtable.ref(cuco::find), num_inserted[0]);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    // compare inserted kv
    
    auto tuple_iter =
        thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(),    insert_values.begin(), 
                                                     contained_keys.begin(), contained_values.begin()));
    // Iterate over all slot contents and verify that `slot.key + 1 == slot.value` is always true.

    cout << "Starting thrust::all_of\n";
    auto result = thrust::all_of(
        thrust::device, tuple_iter, tuple_iter + num_inserted[0], [] __device__(auto const& tuple) {
        return thrust::get<0>(tuple) == thrust::get<2>(tuple) && *thrust::get<1>(tuple) == *thrust::get<3>(tuple); // thrust::get<1>(tuple).rho
    });

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    if (result) { cout << "Success! Target values are properly retrieved.\n"; } //incremented
    else { cout << "Failed at comparison\n"; } //incremented
}

int main() {
    try {
        test_gradients_baseGrid();
    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
