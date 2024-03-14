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
void checkLast(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

// convert from transposed Hilbert index to Hilbert index
void transposeToHilbert(const int X[NDIM], const int L, int &hindex) {
    int n = 0;
    hindex = 0;
    for (short i = 0; i < NDIM; ++i) {
        for (int b = 0; b < L; ++b) {
            n = (b * NDIM) + i;
            hindex |= (((X[NDIM-i-1] >> b) & 1) << n);
        }
    }
}

// convert from Hilbert index to transposed Hilbert index
void hilbertToTranspose(const int hindex, const int L, int (&X)[NDIM]) {
    int h = hindex;
    for (short i = 0; i < NDIM; ++i) X[i] = 0;
    for (short i = 0; i < NDIM * L; ++i) {
        short a = (NDIM - (i % NDIM) - 1);
        X[a] |= (h & 1) << (i / NDIM);
        h >>= 1;
    }
}

// compute the Hilbert index for a given 4-idx (i, j, k, L)
void getHindex(idx4 idx_cell, int& hindex) {
    int X[NDIM];
    for (int i=0; i<NDIM; i++){
        X[i] = idx_cell.idx3[i];
    }
    int L = idx_cell.L;
    int m = 1 << (L - 1), p, q, t;
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
void getHindexInv(int hindex, int L, idx4& idx_cell) {
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
/*double rhoFunc(const double coordMid[NDIM], const double cellSide, const double sigma) {
    int n = round((cellSide - STEP_EPS)/STEP_EPS);
    double eps = (cellSide - STEP_EPS)/n;

    double coord[NDIM], coordEnd[NDIM], coordStart[NDIM];
    for (int i = 0; i < NDIM; i++) {
        coordStart[i] = coordMid[i] - cellSide/2.0 + eps/2.0; // center of little cell
        coordEnd[i]   = coordMid[i] + cellSide/2.0 - eps/2.0;
    }

    double res = 0.0;
    for (int i = 0; i < pow(NDIM, n) + 1; i++) {
        double rsq = 0;
        for (short j = 0; j < NDIM; j++) {
            rsq += pow(coord[j] - 0.5, 2);
        }
        double rho = exp(-rsq / (2 * sigma)) / pow(2 * M_PI * sigma*sigma, 1.5);
        res += rho;

        if (i == pow(NDIM, n)) {
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
            } else
                throw runtime_error("impossible: went out of the upper right corner");
        }
        coord[idx] += eps;
    }
    
    return res / pow(NDIM, n);
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

// find a cell by 4-index in the hashtable
// GPU version: use map_view_type's find function
/*Cell* find(map_type& hashtable, const idx4& idx_cell) {
    thrust::device_vector<idx4> key;
    thrust::device_vector<Cell*> value(1);
    key.push_back(idx_cell);
    hashtable.find(key.begin(), key.end(), value.begin()); //, ramses_hash{}, idx4_equals{}
    return value[0];
}*/
template <typename Map>
__device__ void find(Map hashtable, const idx4 idx_cell, Cell *pCell) {
    auto pair = hashtable.find(idx_cell);
    pCell = pair->second;
}

// check if a cell exists by 4-index
/*bool keyExists(const idx4& idx_cell, map_type &hashtable) {
    Cell* pCell = find(hashtable, idx_cell);
    return pCell != empty_pcell_sentinel;
}*/
bool keyExists(const idx4& idx_cell, host_map &host_table) {
    return host_table.find(idx_cell) != host_table.end();
}
template <typename Map>
__device__ void keyExists(const idx4 idx_cell, Map hashtable, bool &res) {
    Cell* pCell = nullptr;
    find(hashtable, idx_cell, pCell);
    res = pCell != empty_pcell_sentinel;
}

// insert a cell into the hashtable
/*void insert(map_type &hashtable, const idx4& key, Cell* const value) {
    thrust::device_vector<idx4> insert_keys;
    thrust::device_vector<Cell*> insert_values;
    insert_keys.push_back(key);
    insert_values.push_back(value);
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));
    hashtable.insert(zipped, zipped + insert_keys.size());
}*/

// print hash table index
template <typename Map>
void printHashtableIdx(SizeMap<Map>& sizeTable) {
    size_t numCells = sizeTable.numCells;
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    sizeTable.hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    sizeTable.hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin())); //, ramses_hash{}, idx4_equals{}

    thrust::device_vector<thrust::tuple<idx4, Cell*>> entries(numCells);
    for (auto it = zipped; it != zipped + numCells; it++) {
        entries[it - zipped] = *it;
    }
    idx4 idx_cell;
    Cell* pCell = nullptr;

    cout << "CELLS\n";
    for (auto entry : entries) { // entry is on device
        thrust::tuple<idx4, Cell*> t = entry; // t is on host
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        if (idx_cell.idx3[0] == 25 && idx_cell.idx3[1] == 32)
            cout << idx_cell << ' ' << pCell->rho << ' ';
    }
    cout << endl;
}

void printHashtableIdx(host_map &host_table) {
    cout << "CELLS\n";
    for (auto kv : host_table) {
        if (kv.first.idx3[0] == 25 && kv.first.idx3[1] == 32)
            cout << kv.first << ' ' << kv.second.rho << ' ';
    }
    cout << endl;
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
__device__ void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, Map hashtable) {
    idx4 idx_neighbor;
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    keyExists(idx_neighbor, hashtable, exists); 
    is_notref = !exists && !is_border;
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
    Cell* pCell = nullptr;
    find(hashtable, idx_neighbor, pCell);
    rho_neighbor = pCell->rho * int(!is_border) + rho_boundary * int(is_border);
}

// compute the gradient for one cell
template <typename Map>
__device__ void calcGradCell(const idx4 idx_cell, Cell* cell, Map hashtable) {
    bool is_ref[2];
    double dx, rho[3];
    int fd_case;
    dx = pow(0.5, idx_cell.L);
    rho[2] = cell->rho;
    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            getNeighborInfo(idx_cell, dir, pos, is_ref[pos], rho[pos], hashtable);
        }
        fd_case = is_ref[0] + 2 * is_ref[1];
        cell->rho_grad[dir] = (FD_KERNEL[fd_case][0] * rho[0] + FD_KERNEL[fd_case][1] * rho[2] + FD_KERNEL[fd_case][2] * rho[1]) / (FD_KERNEL[fd_case][3] * dx);
    }
}

// compute the gradient
template <typename Map>
__global__ void calcGrad(Map hashtable, auto zipped, size_t numCells) {
    idx4 idx_cell;
    Cell* pCell = nullptr;
    for (auto it = zipped; it != zipped + numCells; it++) {
        thrust::tuple<idx4, Cell*> t = *it;
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        calcGradCell(idx_cell, pCell, hashtable);
    }
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
void makeBaseGrid(Cell (&host_grid)[NMAX], host_map &host_table) {
    cout << "HELLO0 makeBaseGrid\n";
    printf("HELLO makeBaseGrid\n");
    idx4 idx_cell;
    printf("HELLO2 makeBaseGrid\n");
    for (int L = 0; L <= LBASE; L++) {
        for (int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            printf("ENTERING getHindexInv\n");
            getHindexInv(hindex, L, idx_cell);
            printf("ENTERING setGridCell\n");
            setGridCell(host_grid, idx_cell, hindex, L == LBASE, host_table); // cells have flag_leaf == 1 at L == LBASE == 3
        }
    }
};

// set a grid cell in the grid array and the hash table
void setGridCell(Cell (&host_grid)[NMAX], const idx4 idx_cell, const int hindex, int32_t flag_leaf,
                 host_map &host_table) {
    if (keyExists(idx_cell, host_table)) throw runtime_error("setting existing cell");

    int offset;
    double dx, coord[3];
    offset = (pow(2, NDIM * idx_cell.L) - 1) / (pow(2, NDIM) - 1);
    dx = 1.0 / pow(2, idx_cell.L);
    for (int i = 0; i < NDIM; i++) {
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
    }
    idx_cell.println();
    host_grid[offset + hindex].rho = rhoFunc(coord, sigma); // , dx
    host_grid[offset + hindex].flag_leaf = flag_leaf;
    if (offset + hindex >= NMAX) throw runtime_error("offset () + hindex >= N_cell_max");
    
    host_table[idx_cell] = host_grid[offset + hindex];
    printf("HOST ");
    idx_cell.print();
    host_grid[offset + hindex].print();
    printf("\n");
}

// refine the grid by one level
void refineGrid1lvl(Cell (&host_grid)[NMAX], host_map &host_table) {
    for (auto kv : host_table) {
        if (refCrit(kv.second.rho) && kv.second.flag_leaf) {
            refineGridCell(host_grid, kv.first, host_table);
        }
    }
}

// set child cells in the grid array and hash table
void setGridChildren(Cell (&host_grid)[NMAX], idx4 idx_cell, short i, 
                       host_map &host_table) {
    if (i == NDIM) {
        int hindex;
        getHindex(idx_cell, hindex);
        setGridCell(host_grid, idx_cell, hindex, 1, host_table);
        return;
    }
    setGridChildren(host_grid, idx_cell, i+1, host_table);
    idx_cell.idx3[i]++;
    setGridChildren(host_grid, idx_cell, i+1, host_table);
}

// refine a grid cell
void refineGridCell(Cell (&host_grid)[NMAX], const idx4 idx_cell, host_map &host_table) {
    int hindex;
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
    setGridChildren(host_grid, idx_child, 0, host_table);
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
            refineGridCell(host_grid, idx_neighbor, host_table);
        }
    }
}

template <typename ValueIter>
__global__ void insert_vector_pointers(ValueIter insert_values_begin, 
                                       Cell* pointer_underl_values_begin, 
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

        if (map_ref.insert(cuco::pair{key_begin[tid], value_begin[tid]})) {
            ++counter;  // Count number of successfully inserted keys
            if (idx_cell == idx4{0, 0, 0, 0}) {
                printf("INSERTING: ");
            }
        } else {
            if (idx_cell == idx4{0, 0, 0, 0}) {
                printf("CANNOT INSERT: ");
            }
        }
        if (idx_cell == idx4{0, 0, 0, 0}) {
            idx_cell.print();
            printf(" ");
            pCell->println();
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
    Cell host_grid[NMAX];
    host_map host_table;
    
    cout << "Making base grid" << endl;
    
    makeBaseGrid(host_grid, host_table);

    /*const int num_ref = LMAX - LBASE;
    cout << "Refining grid levels" << endl;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(host_grid, host_table);
    }
    cout << "Finished refining grid levels" << endl;*/
    //string filename = "grid-host.csv";
    writeGrid(host_table, "grid-host.csv");
    

    // hashtable insert values from host_grid
    auto constexpr block_size = 256;
    auto const grid_size      = (NMAX + block_size - 1) / block_size;

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
    insert_vector_pointers<<<grid_size, block_size>>>(insert_values.begin(), 
                                                      thrust::raw_pointer_cast(underl_values.data()),
                                                      host_table.size(),
                                                      num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

    
    auto hashtable = cuco::static_map{cuco::extent<std::size_t, NMAX>{},
                                      cuco::empty_key{empty_idx4_sentinel},
                                      cuco::empty_value{empty_pcell_sentinel},
                                      thrust::equal_to<idx4>{},
                                      cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
    auto insert_ref = hashtable.ref(cuco::insert);

    
    // reset num_inserted
    num_inserted[0] = 0;
    insert<<<grid_size, block_size>>>(insert_ref,
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

    //printf("\n\ninsert_keys\n");

    //printHashtable<<<1, 1>>>(insert_keys.begin(), insert_values.begin(), num_inserted[0]);

    //cudaDeviceSynchronize();
    //CHECK_LAST_CUDA_ERROR();
    //printf("\n\ncontained_keys\n");

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

    /*cout << "Calculating gradients" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU
    //map_view_type view = hashtable.get_device_view();
    // get zipped values before kicking off kernels
    size_t numCells = sizeTable.numCells;
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    sizeTable.hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    sizeTable.hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin())); //, ramses_hash{}, idx4_equals{}
    
    auto hashtable_find_ref = sizeTable.hashtable.ref(cuco::find);
    calcGrad<<<1, 1>>>(hashtable_find_ref, zipped, numCells);

    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    writeGrid(sizeTable);*/
}

int main() {
    test_makeBaseGrid();
}
