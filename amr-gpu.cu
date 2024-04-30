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
void getHindex(idx4 idx_oct, long int &hindex) {
    int X[NDIM];
    for (int i=0; i<NDIM; i++){
        X[i] = idx_oct.idx3[i];
    }
    short int L = idx_oct.L;
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
void getHindexInv(long int hindex, int L, idx4& idx_oct) {
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
        idx_oct.idx3[i] = X[i];
    }
    idx_oct.L = L;
}

__host__ __device__ void getIndexInv(long int index, int L, idx4 &idx_oct) {
    for (int dim = 0; dim < NDIM; dim++) {
        long int scale = (long int)pow(2, (NDIM-1-dim)*L);
        idx_oct.idx3[dim] = int(index / scale);
        index %= scale;
    }
    idx_oct.L = L;
}

__host__ __device__ void getIndex(const idx4 idx_oct, long int &index) {
    long int scale = 1;
    index = 0;
    for (int dim = NDIM-1; dim >= 0; dim--) {
        index += scale * idx_oct.idx3[dim];
        //printf("index %ld scale %ld\n", index, scale);
        scale *= int(pow(2, idx_oct.L));
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

// compute the index of the parent oct
void getParentIdx(const idx4 &idx_oct, idx4 &idx_parent) {
    for (short i = 0; i < NDIM; i++) {
        idx_parent.idx3[i] = idx_oct.idx3[i] / 2;
    }
    idx_parent.L = idx_oct.L - 1;
}

// compute the indices of the neighbor octs on a given face
__host__ __device__ void getNeighborIdx(const idx4 idx_oct, const int dir, const bool pos, idx4 &idx_neighbor) {
    // after this getNeighborIdx is applied, must check if neighbor exists (border) !!!
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor.idx3[i] = idx_oct.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_oct.L;
}

// check if a given face is a border of the computational domain
__host__ __device__ void checkIfBorder(const idx4 &idx_oct, const int dir, const bool pos, bool &is_border) {
    is_border = idx_oct.idx3[dir] == int(pos) * (pow(2, idx_oct.L) - 1);
}

template <typename Map>
__device__ void keyExists(const idx4 idx_oct, Map hashtable_ref, bool &res) {
    res = hashtable_ref.find(idx_oct) != hashtable_ref.end();
}

template <typename Map>
__device__ void getNeighborInfo(const idx4 idx_oct, const int dir, const bool pos, 
                                bool &is_ref, float &rho_neighbor, Map hashtable_ref) {
    idx4 idx_neighbor;
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the oct is a border oct
    checkIfBorder(idx_oct, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_oct, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the oct is not a border oct, then the neighbor is not refined
    keyExists(idx_neighbor, hashtable_ref, exists); 
    is_notref = !exists && !is_border;
    is_ref = !is_notref && !is_border;
    // if the oct is a border oct, set the neighbor index to the oct index (we just want a valid key for the hashtable)
    // if the neighbor is not refined, set the neighbor index to the index of the parent octs neighbor
    // if the neighbor is refined, don't change the neighbor index

    if (is_notref)
        printf("is_notref %d ni %d nj %d nk %d exists %d is_border %d  i %d j %d k %d\n", is_notref, idx_neighbor.idx3[0], 
            idx_neighbor.idx3[1], idx_neighbor.idx3[2], exists, is_border, idx_oct.idx3[0], idx_oct.idx3[1], 
            idx_oct.idx3[2]);

    for (short i = 0; i < NDIM; i++) {
        idx1_parent_neighbor = idx_oct.idx3[i] / 2 + (int(pos) * 2 - 1) * int(i == dir);
        idx_neighbor.idx3[i] = idx_oct.idx3[i] * int(is_border) + idx_neighbor.idx3[i] * int(is_ref) 
                                + idx1_parent_neighbor * int(is_notref);
    }
    // subtract one from the AMR level if the neighbor is not refined
    idx_neighbor.L = idx_oct.L - int(is_notref);

    // if the oct is a border oct, use the boundary condition
    Oct* pOct = hashtable_ref.find(idx_neighbor)->second;
    rho_neighbor = pOct->rho * int(!is_border) + rho_boundary * int(is_border);

    /*if (idx_oct == idx4(1, 1, 1, 2))
        printf("neighbor [%d, %d, %d](L=%d) %d %f %f\n", idx_neighbor.idx3[0], idx_neighbor.idx3[1], 
            idx_neighbor.idx3[2], idx_neighbor.L, is_border, rho_boundary, pOct->rho);*/
}

// compute the gradient for one oct
template <typename Map>
__device__ void calcGradOct(const idx4 idx_oct, Oct* oct, Map hashtable_ref) {
    bool is_ref[2];
    // explicitly use NDIM == 3
    double dx;
    float rho[3];
    int fd_case;

    dx = pow(0.5, idx_oct.L);
    rho[2] = oct->rho;

    /*if (idx_oct == idx4(1, 1, 1, 2)) {
        printf("\n");
    }*/

    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_oct, dir, pos, is_ref[pos], rho[pos], hashtable_ref);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        oct->rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2]
                             + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
        /*if (idx_oct == idx4(1, 1, 1, 2)) {
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
        idx4 idx_oct = *contained_keys;
        Oct *pOct = hashtable_ref.find(idx_oct)->second;
        calcGradOct(idx_oct, pOct, hashtable_ref);

        contained_keys += gridDim.x * blockDim.x;
        tid += gridDim.x * blockDim.x;
    }
}

template <typename KeyIter, typename ValueIter>
void writeGrid(KeyIter keys_iter, ValueIter underl_values_iter, size_t num_keys, string filename) {
    // save i, j, k, L, rho, gradients for all octs (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf";
    for (int j = 0; j < 8; j++) {
        outfile << ",rho" << j << ",rho_grad_x" << j << ",rho_grad_y" << j << ",rho_grad_z" << j;
    }
    outfile << "\n";
    for (int i = 0; i < num_keys; i++) {
        idx4 idx_oct = *keys_iter;
        Oct oct = *underl_values_iter;

        outfile << idx_oct.idx3[0] << "," << idx_oct.idx3[1] << "," << idx_oct.idx3[2]
                << "," << idx_oct.L;
        
        for (int j = 0; j < 8; j++) {
            outfile << "," << oct.flag_leaf[j] << "," << oct.rho[j] << "," << oct.rho_grad[j][0]
                << "," << oct.rho_grad[j][1] << "," << oct.rho_grad[j][2];       
        }
        outfile << "\n";

        keys_iter++;
        underl_values_iter++;
    }
    outfile.close();
}

void writeGrid(thrust::device_vector<idx4> &insert_keys, thrust::device_vector<Oct> &insert_vals, string filename) {
    // save i, j, k, L, rho, gradients for all octs (use the iterator) to a file
    ofstream outfile;
    outfile.open(filename);
    outfile << "i,j,k,L,flag_leaf";
    for (int j = 0; j < 8; j++) {
        outfile << ",rho" << j << ",rho_grad_x" << j << ",rho_grad_y" << j << ",rho_grad_z" << j;
    }
    outfile << "\n";
    for (int i = 0; i < insert_keys.size(); i++) {
        idx4 idx_oct = insert_keys[i];
        Oct oct = insert_vals[i];

        outfile << idx_oct.idx3[0] << "," << idx_oct.idx3[1] << "," << idx_oct.idx3[2]
                << "," << idx_oct.L;
        
        for (int j = 0; j < 8; j++) {
            outfile << "," << oct.flag_leaf[j] << "," << oct.rho[j] << "," << oct.rho_grad[j][0]
                << "," << oct.rho_grad[j][1] << "," << oct.rho_grad[j][2];       
        }
        outfile << "\n";
    }
    outfile.close();
}

template <typename KeyIter, typename ValueIter>
__global__ void make1lvlGrid(KeyIter insert_keys_it, ValueIter insert_vals_it, int L, size_t num_inserted, 
                             bool to_offset) {
    long int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //constexpr uint32_t NOCT_MAX_DEVICE[10] = {2, 16, 128, 1024, 8192, 65536, 524288, 4194304, 
    //    33554432, 268435456};

    while (tid < num_inserted) {
        long int index = tid;
        long int offset = 0;
        if (to_offset)
            offset = (pow(2, NDIM * L) - 1) / (pow(2, NDIM) - 1);
        idx4 idx_oct;

        getIndexInv(index, L, idx_oct);
        insert_keys_it[index + offset] = idx_oct;
        setGridOct(idx_oct, index, true, insert_vals_it, num_inserted, to_offset); // octs have flag_leaf == 1 at L == lbase == 3
        
        tid += gridDim.x * blockDim.x;
    }
}

// set a grid oct in the grid array and the hash table
template <typename DevicePtr>
__device__ void setGridOct(const idx4 idx_oct, const long int index, int32_t flag_leaf,
                DevicePtr insert_vals_it, uint32_t num_inserted, bool to_offset=true) {
    long int offset = 0;
    if (to_offset)
        offset = (pow(2, NDIM * idx_oct.L) - 1) / (pow(2, NDIM) - 1);
    // explicitly use NDIM == 3
    double dx, coord[3];
    dx = 1.0 / pow(2, idx_oct.L);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_oct.idx3[i] * dx + dx / 2;
    }

    if (offset + index >= num_inserted) 
        printf("ERROR: offset + index >= NOCT_MAX; offset %ld index %ld NOCT_MAX %u\n", offset, index, num_inserted);
    Oct oct = *(insert_vals_it + offset + index);
    if (!(oct == Oct())) printf("ERROR setting existing oct\n");
    
    insert_vals_it[offset + index] = Oct(rhoFunc(coord, sigma), 0.0, 0.0, 0.0, flag_leaf);
    //printf("HOST ");
    //idx_oct.print();
    //insert_vals_it[offset + index].print();
    //printf("\n");
}

/*// refine the grid by one level
// unsynced with getHindex
void refineGrid1lvl(host_map &host_table) {
    for (auto kv : host_table) {
        if (refCrit(kv.second.rho) && kv.second.flag_leaf) {
            refineGridOct(kv.first, host_table);
        }
    }
}

// set child octs in the grid array and hash table
// unsynced with getHindex
void setGridChildren(idx4 idx_oct, short i, 
                       host_map &host_table) {
    if (i == NDIM) {
        long int hindex;
        getHindex(idx_oct, hindex);
        setGridOct(idx_oct, hindex, 1, host_table);
        return;
    }
    setGridChildren(idx_oct, i+1, host_table);
    idx_oct.idx3[i]++;
    setGridChildren(idx_oct, i+1, host_table);
}

// refine a grid oct
// unsynced with getHindex
void refineGridOct(const idx4 idx_oct, host_map &host_table) {
    long int hindex;
    getHindex(idx_oct, hindex);
    if (!keyExists(idx_oct, host_table)) throw runtime_error("Trying to refine non-existant oct! "+idx_oct.str());
    Oct oct = host_table[idx_oct];
    if (oct.flag_leaf) throw runtime_error("trying to refine non-leaf");
    if (id_oct.L == LMAX) throw runtime_error("trying to refine at max level");
    // make this oct a non-leaf
    oct.flag_leaf = 0;
    idx4 idx_child(idx_oct.idx3, size_t(idx_oct.L + 1));
    for (short dir = 0; dir < NDIM; dir++) idx_child.idx3[dir] *= 2;
    // and create 2^NDIM leaf children
    setGridChildren(idx_child, 0, host_table);
    // refine neighbors if needed
    idx4 idx_neighbor, idx_parent;
    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            bool is_border;
            checkIfBorder(idx_oct, dir, pos, is_border);
            if (is_border) continue;
            getNeighborIdx(idx_oct, dir, pos, idx_neighbor);
            if (keyExists(idx_neighbor, host_table)) continue;
            // we assume that L is at most different by 1
            getParentIdx(idx_oct, idx_parent);
            if (!keyExists(idx_parent, host_table))
                throw runtime_error("idx_parent does not exist! "+idx_parent.str()+' '+idx_oct.str());
            getNeighborIdx(idx_parent, dir, pos, idx_neighbor);
            if (!keyExists(idx_neighbor, host_table)) continue; // parent is at border
            refineGridOct(idx_neighbor, host_table);
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
        thrust::tuple<idx4, Oct*> t = thrust::make_tuple(key_begin[tid], value_begin[tid]);
        idx4 idx_oct = t.get<0>();
        Oct *pOct = t.get<1>();

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
        idx4 idx_oct = t.get<0>();
        Oct* pOct = hashtable_ref.find(idx_oct)->second; //t.get<1>();
        
        printf("GPU ");
        idx_oct.print();
        if (!pOct) printf("ERROR: accessing null oct ptr");
        pOct->print();
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
    int32_t num_octs = 0;
    for (int L = 0; L < LBASE; L++) {
        num_octs += pow(2, NDIM * L);
    }
    thrust::device_vector<idx4> insert_keys(num_octs);
    thrust::device_vector<Oct> underl_values(num_octs);
    
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

    thrust::device_vector<Oct*> insert_values(num_octs);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      num_octs,
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<uint32_t, 16*NOCT_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_poct_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                      insert_keys.begin(),
                                      insert_values.begin(),
                                      num_octs,
                                      num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;


    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Oct*> contained_values(num_inserted[0]);
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
    int32_t num_octs = pow(2, NDIM * LBASE);
    
    cout << "Making base grid" << endl;
    thrust::device_vector<idx4> insert_keys(num_octs);
    thrust::device_vector<Oct> underl_values(num_octs);
    
    make1lvlGrid<<<GRID_SIZE, BLOCK_SIZE>>>(insert_keys.begin(),
                                            underl_values.begin(), 
                                            LBASE,
                                            num_octs,
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

    thrust::device_vector<Oct*> insert_values(num_octs);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      num_octs,
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, 16*NOCT_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_poct_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<GRID_SIZE, BLOCK_SIZE>>>(insert_ref,
                                      insert_keys.begin(),
                                      insert_values.begin(),
                                      num_octs,
                                      num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;

    
    thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Oct*> contained_values(num_inserted[0]);
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
                        int32_t num_octs, int repeat=1) {
    auto start = high_resolution_clock::now();

    thrust::device_vector<int> num_inserted(1);
    
    auto hashtable = cuco::static_map{cuco::extent<uint32_t, 2*NOCT_MAX>{},
                                    cuco::empty_key{empty_idx4_sentinel},
                                    cuco::empty_value{empty_poct_sentinel},
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
                                    num_octs,
                                    num_inserted.data().get());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    cout << "time for GPU insert: " << duration.count()/1000.0 << " ms" << endl;

    /*thrust::device_vector<idx4> contained_keys(num_inserted[0]);
    thrust::device_vector<Oct*> contained_values(num_inserted[0]);
    // this is random ordered and is DIFFERENT from insert_keys order
    hashtable.retrieve_all(contained_keys.begin(), contained_values.begin());

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
    
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    cout << "time for retrieve_all: " << duration.count()/1000.0 << " ms" << endl;*/

    auto grid_size = (NOCT_MAX_ARR[L] + block_size - 1) / block_size;
    printf("num threads: %ld\n", grid_size*block_size);

    double avg_time = 0;
    for (int i = 0; i < repeat; i++) {
        auto start = high_resolution_clock::now();

        // run as kernel on GPU

        calcGrad<<<grid_size, block_size>>>(find_ref,
                                            insert_keys_it,
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

        auto grid_size = (NOCT_MAX_ARR[L] + block_size - 1) / block_size;
        int32_t num_octs = pow(2, NDIM * L);
        
        thrust::device_vector<idx4> insert_keys(num_octs);
        thrust::device_vector<Oct> underl_values(num_octs);
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after creating insert_keys and underl_values: " << duration.count()/1000.0 << " ms" << endl;

        make1lvlGrid<<<grid_size, block_size>>>(insert_keys.begin(),
                                                underl_values.begin(), 
                                                L,
                                                num_octs,
                                                false);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "Time after GPU make1lvlGrid: " << duration.count()/1000.0 << " ms" << endl;


        thrust::device_vector<Oct*> insert_values(num_octs);

        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        start = high_resolution_clock::now();
        cout << "time after creating insert_values: " << duration.count()/1000.0 << " ms" << endl;

        thrust::device_vector<int> num_inserted(1);
        insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                        thrust::raw_pointer_cast(underl_values.data()),
                                                        num_octs,
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
                                        num_octs, 
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
    thrust::device_vector<Oct> underl_values(1);
    thrust::device_vector<Oct*> insert_values(1);

    insert_keys[0] = idx4(0, 0, 0, 0);
    underl_values[0] = Oct(1000.00, 0.00, 0.00, 0.00, 1);

    thrust::device_vector<int> num_inserted(1);
    insert_vector_pointers<<<GRID_SIZE, BLOCK_SIZE>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      1,
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<uint32_t, 16*NOCT_MAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_poct_sentinel},
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
    thrust::device_vector<Oct*> contained_values(num_inserted[0]);
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
        /*if (NOCT_MAX != lround(2*pow(2, LMAX*NDIM))) {
            throw runtime_error("NOCT_MAX != 2*2^(LMAX*NDIM); NOCT_MAX "+to_string(NOCT_MAX)+" 2*2^(LMAX*NDIM) "
                +to_string(lround(2*pow(2, LMAX*NDIM))));
        }*/

        test_speed();
        
    } catch  (const runtime_error& error) {
        printf(error.what());
    }
}
