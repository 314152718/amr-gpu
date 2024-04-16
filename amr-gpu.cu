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
        for (short i = NDIM - 1; i >= 0; i--) {
            if (X[i] & q) { // invert
                X[0] ^= p;
            } else {
                t = (X[0]^X[i]) & p;
                X[0] ^= t;
                X[i] ^= t;
            }
        } 
    } // exchange
    for (int i=0; i<NDIM; i++) {
        idx_cell.idx3[i] = X[i];
    }
    idx_cell.L = L;
}

__host__ __device__ void getIndexInv(long int index, int L, idx4 &idx_cell) {
    for (int dim = 0; dim < NDIM; dim++) {
        long int scale = (long int)pow(2, (NDIM-1-dim)*L);
        idx_cell.idx3[dim] = int(index / scale);
        index %= scale;
    }
    idx_cell.L = L;
}

__host__ __device__ void getIndex(const idx4 idx_cell, long int &index) {
    long int scale = 1;
    index = 0;
    for (int dim = NDIM-1; dim >= 0; dim--) {
        index += scale * idx_cell.idx3[dim];
        //printf("index %ld scale %ld\n", index, scale);
        scale *= int(pow(2, idx_cell.L));
    }
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
__device__ double rhoFunc(const double coord[NDIM], const double sigma) {
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

template <typename Map>
__device__ void keyExists(const idx4 idx_cell, Map hashtable_ref, bool &res) {
    res = hashtable_ref.find(idx_cell) != hashtable_ref.end();
}

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

    if (is_notref)
        printf("is_notref %d ni %d nj %d nk %d exists %d is_border %d  i %d j %d k %d\n", is_notref, idx_neighbor.idx3[0], 
            idx_neighbor.idx3[1], idx_neighbor.idx3[2], exists, is_border, idx_cell.idx3[0], idx_cell.idx3[1], 
            idx_cell.idx3[2]);

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

    /*if (idx_cell == idx4(1, 1, 1, 2))
        printf("neighbor [%d, %d, %d](L=%d) %d %f %f\n", idx_neighbor.idx3[0], idx_neighbor.idx3[1], 
            idx_neighbor.idx3[2], idx_neighbor.L, is_border, rho_boundary, pCell->rho);*/
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

    /*if (idx_cell == idx4(1, 1, 1, 2)) {
        printf("\n");
    }*/

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_cell, dir, pos, is_ref[pos], rho[pos], hashtable_ref);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell->rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                             + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
        /*if (idx_cell == idx4(1, 1, 1, 2)) {
            printf("%d %f %f %f\n", dir, rho[0], rho[1], rho[2]);
        }*/
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

void writeGrid(thrust::device_vector<idx4> &insert_keys, thrust::device_vector<Cell> &insert_vals, string filename) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    for (int i = 0; i < insert_keys.size(); i++) {
        idx4 idx_cell = insert_keys[i];
        Cell cell = insert_vals[i];

        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << cell.flag_leaf << "," << cell.rho << "," << cell.rho_grad[0]
                << "," << cell.rho_grad[1] << "," << cell.rho_grad[2] << "\n";
    }
    outfile.close();
}

template <typename KeyIter, typename ValueIter>
__global__ void make1lvlGrid(KeyIter insert_keys_it, ValueIter insert_vals_it, int L, size_t num_inserted, 
                             bool to_offset) {
    long int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    while (tid < num_inserted) {
        int index = tid;
        int offset = 0;
        if (to_offset)
            offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
        idx4 idx_cell;

        getIndexInv(index, L, idx_cell);
        insert_keys_it[index + offset] = idx_cell;
        setGridCell(idx_cell, index, true, insert_vals_it, to_offset); // cells have flag_leaf == 1 at L == lbase == 3
        
        tid += gridDim.x * blockDim.x;
    }
}

// set a grid cell in the grid array and the hash table
template <typename DevicePtr>
__device__ void setGridCell(const idx4 idx_cell, const long int index, int32_t flag_leaf,
                DevicePtr insert_vals_it, bool to_offset=true) {
    int offset = 0;
    if (to_offset)
        offset = (pow(2, NDIM * idx_cell.L) - 1) / (pow(2, NDIM) - 1);
    // explicitly use NDIM == 3
    double dx, coord[3];
    dx = 1.0 / pow(2, idx_cell.L);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
    }

    if (offset + index >= NCELL_MAX_ARR[idx_cell.L]) printf("ERROR: offset + index >= NCELL_MAX\n");
    Cell cell = *(insert_vals_it + offset + index);
    if (!(cell == Cell())) printf("ERROR setting existing cell\n");
    
    insert_vals_it[offset + index] = Cell(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, flag_leaf);
    //printf("HOST ");
    //idx_cell.print();
    //insert_vals_it[offset + index].print();
    //printf("\n");
}

/*// refine the grid by one level
// unsynced with getHindex
void refineGrid1lvl(host_map &host_table) {
    for (auto kv : host_table) {
        if (refCrit(kv.second.rho) && kv.second.flag_leaf) {
            refineGridCell(kv.first, host_table);
        }
    }
}

// set child cells in the grid array and hash table
// unsynced with getHindex
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
// unsynced with getHindex
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
}*/

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
    //printf("%lu\n", num_keys);
 
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
__global__ void printHashtable(KeyIter key_iter, Map hashtable_ref, int32_t num_keys) {
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
    int32_t num_cells = 0;
    for (int L = 0; L < LBASE; L++) {
        num_cells += pow(2, NDIM * L);
    }
    thrust::device_vector<idx4> insert_keys(num_cells);
    thrust::device_vector<Cell> underl_values(num_cells);
    
    cout << "Making base grid" << endl;
    for (int L = 0; L < LBASE; L++) {
        int32_t num_inserted = pow(2, NDIM * L);
        
        make1lvlGrid<<<GRID_SIZE, BLOCK_SIZE>>>(insert_keys.begin(),
                                                underl_values.begin(), 
                                                LBASE,
                                                num_inserted,
                                                true);
    }
    writeGrid(insert_keys, underl_values, "grid-host.csv");
    

    // hashtable insert values from host_table

    thrust::device_vector<Cell*> insert_values(num_cells);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      num_cells,
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
                                      num_cells,
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
    int32_t num_cells = pow(2, NDIM * LBASE);
    
    cout << "Making base grid" << endl;
    thrust::device_vector<idx4> insert_keys(num_cells);
    thrust::device_vector<Cell> underl_values(num_cells);
    
    make1lvlGrid<<<GRID_SIZE, BLOCK_SIZE>>>(insert_keys.begin(),
                                            underl_values.begin(), 
                                            LBASE,
                                            num_cells,
                                            false);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    /*const int num_ref = LMAX - LBASE;
    cout << "Refining grid levels" << endl;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(host_table);
    }
    cout << "Finished refining grid levels" << endl;*/
    //string filename = "grid-host.csv";
    writeGrid(insert_keys, underl_values, "grid-host.csv");
    

    // hashtable insert values from host_table

    thrust::device_vector<Cell*> insert_values(num_cells);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      num_cells,
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
                                      num_cells,
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


template <typename KeyIter, typename ValueIter>
double time_calcGrad(int block_size, int L,
                        KeyIter insert_keys_it, ValueIter insert_vals_it, 
                        int32_t num_cells, int repeat=1) {
    auto start = high_resolution_clock::now();

    thrust::device_vector<int> num_inserted(1);
    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NCELL_MAX>{},
                                    cuco::empty_key{empty_idx4_sentinel},
                                    cuco::empty_value{empty_pcell_sentinel},
                                    thrust::equal_to<idx4>{},
                                    cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);
    auto find_ref = hashtable.ref(cuco::find);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    cout << "time after creating hashtable and ref: " << duration.count()/1000.0 << " ms" << endl;

    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                    insert_keys_it,
                                    insert_vals_it,
                                    num_cells,
                                    num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    cout << "time for GPU insert: " << duration.count()/1000.0 << " ms" << endl;

    
    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Cell*> contained_values(num_inserted[0]);
    // this is random ordered and is DIFFERENT from insert_keys order
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    cout << "time for retrieve_all: " << duration.count()/1000.0 << " ms" << endl;

    auto grid_size = (NCELL_MAX_ARR[L] + block_size - 1) / block_size;
    printf("num threads: %ld\n", grid_size*block_size);

    double avg_time = 0;
    for (int i = 0; i < repeat; i++) {
        auto start = high_resolution_clock::now();

        // run as kernel on GPU

        calcGrad<<<grid_size, block_size>>>(find_ref, 
                                            contained_keys.begin(), 
                                            num_inserted[0]); // add for (50)
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

void test_speed() {
    auto start_all = high_resolution_clock::now();

    ofstream outfile;
    outfile.open("time_hashtable.csv");
    outfile << "L,dur_ms\n";
    int block_size = 512;

    for (int L = LBASE; L <= LMAX; L++) {
        auto start0 = high_resolution_clock::now();
        auto start = start0;

        auto grid_size = (NCELL_MAX_ARR[L] + block_size - 1) / block_size;
        int32_t num_cells = pow(2, NDIM * L);
        
        thrust::device_vector<idx4> insert_keys(num_cells);
        thrust::device_vector<Cell> underl_values(num_cells);
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after creating insert_keys and underl_values: " << duration.count()/1000.0 << " ms" << endl;

        make1lvlGrid<<<grid_size, block_size>>>(insert_keys.begin(),
                                                underl_values.begin(), 
                                                L,
                                                num_cells,
                                                false);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after GPU make1lvlGrid: " << duration.count()/1000.0 << " ms" << endl;


        thrust::device_vector<Cell*> insert_values(num_cells);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "time after creating insert_values: " << duration.count()/1000.0 << " ms" << endl;

        thrust::device_vector<int> num_inserted(1);
        insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                        thrust::raw_pointer_cast(underl_values.data()),
                                                        num_cells,
                                                        num_inserted.data().get());
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        cout << "time for GPU insert_vector_pointers: " << duration.count()/1000.0 << " ms" << endl;

        cout << "L = " << L << ", block size " << block_size << endl;
        double avg_time = time_calcGrad(block_size, L,
                                        insert_keys.begin(),
                                        insert_values.begin(),
                                        num_cells, 
                                        10);

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
    printf("amr gpu hashtable\n");
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
