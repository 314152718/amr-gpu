// defines
#define _USE_MATH_DEFINES

// cpu includes
#include <iostream>
#include <stdio.h>
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
#include <cub/block/block_reduce.cuh>
#include <cuda/std/atomic>

// namespaces
using namespace std;
using namespace std::chrono;

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
__host__ __device__ void getNeighborIdx(const idx4 idx_cell, const int dir, const bool pos, idx4 idx_neighbor) {
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
Cell* find(map_type& hashtable, const idx4& idx_cell) {
    thrust::device_vector<idx4> key;
    thrust::device_vector<Cell*> value(1);
    key.push_back(idx_cell);
    hashtable.find(key.begin(), key.end(), value.begin());
    return value[0];
}
__device__ void find(map_view_type &hashtable, const idx4 idx_cell, Cell *pCell) {
    cuco::static_map<idx4, Cell *>::device_view::const_iterator pair = hashtable.find(idx_cell);
    pCell = pair->second;
}

// check if a cell exists by 4-index
bool checkIfExists(const idx4& idx_cell, map_type &hashtable) {
    Cell* pCell = find(hashtable, idx_cell);
    return pCell != empty_pcell_sentinel;
}
__device__ void checkIfExists(const idx4 idx_cell, map_view_type &hashtable, bool &res) {
    Cell* pCell = nullptr;
    find(hashtable, idx_cell, pCell);
    res = pCell != empty_pcell_sentinel;
}

// initialize the base level grid
void makeBaseGrid(Cell (&grid)[NMAX], map_type &hashtable) {
    idx4 idx_cell;
    for (int L = 0; L <= LBASE; L++) {
        for (int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            setGridCell(idx_cell, hindex, L == LBASE, hashtable);
        }
    }
};

// set a grid cell in the grid array and the hash table
void setGridCell(const idx4 idx_cell, const int hindex, int32_t flag_leaf, map_type &hashtable) {
    if (checkIfExists(idx_cell, hashtable)) throw runtime_error("setting existing cell");
    int offset;
    double dx, coord[3];
    offset = (pow(2, NDIM * idx_cell.L) - 1) / (pow(2, NDIM) - 1);
    dx = 1 / pow(2, idx_cell.L);
    for (int i = 0; i < NDIM; i++)
        coord[i] = idx_cell.idx3[i] * dx + dx / 2;
    grid[offset + hindex].rho = rhoFunc(coord, sigma);
    grid[offset + hindex].flag_leaf = flag_leaf;
    if (offset + hindex >= NMAX) throw runtime_error("offset () + hindex >= N_cell_max");
    insert(hashtable, idx_cell, &grid[offset + hindex]);
}


// insert a cell into the hashtable
void insert(map_type &hashtable, const idx4& key, Cell* const value) {
    thrust::device_vector<idx4> insert_keys;
    thrust::device_vector<Cell*> insert_values;
    insert_keys.push_back(key);
    insert_values.push_back(value);
    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));
    hashtable.insert(zipped, zipped + insert_keys.size());
}

// set child cells in the grid array and hash table
void setChildrenHelper(idx4 idx_cell, short i, map_type &hashtable) {
    if (i == NDIM) {
        int hindex;
        getHindex(idx_cell, hindex);
        setGridCell(idx_cell, hindex, 1, hashtable);
        return;
    }
    setChildrenHelper(idx_cell, i+1, hashtable);
    idx_cell.idx3[i]++;
    setChildrenHelper(idx_cell, i+1, hashtable);
}

// refine a grid cell
void refineGridCell(const idx4 idx_cell, map_type &hashtable) {
    int hindex;
    getHindex(idx_cell, hindex);
    Cell *pCell = find(hashtable, idx_cell);
    if (pCell == empty_pcell_sentinel) throw runtime_error("Trying to refine non-existant cell! "+idx_cell.str());
    if (!pCell->flag_leaf) throw runtime_error("trying to refine non-leaf");
    if (idx_cell.L == LMAX) throw runtime_error("trying to refine at max level");
    // make this cell a non-leaf
    pCell->flag_leaf = 0;
    idx4 idx_child(idx_cell.idx3, idx_cell.L + 1);
    for (short dir = 0; dir < NDIM; dir++) idx_child.idx3[dir] *= 2;
    // and create 2^NDIM leaf children
    setChildrenHelper(idx_child, 0, hashtable);
    // refine neighbors if needed
    idx4 idx_neighbor, idx_parent;
    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            bool is_border;
            checkIfBorder(idx_cell, dir, pos, is_border);
            if (is_border) continue;
            getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
            if (checkIfExists(idx_neighbor, hashtable)) continue;
            // we assume that L is at most different by 1
            getParentIdx(idx_cell, idx_parent);
            if (!checkIfExists(idx_parent, hashtable))
                throw runtime_error("idx_parent does not exist! "+idx_parent.str()+' '+idx_cell.str());
            getNeighborIdx(idx_parent, dir, pos, idx_neighbor);
            if (!checkIfExists(idx_neighbor, hashtable)) continue; // parent is at border
            refineGridCell(idx_neighbor, hashtable);
        }
    }
}

// print hash table index
void printHashtableIdx(map_type& hashtable) {
    size_t numCells = hashtable.get_size();
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));

    thrust::device_vector<thrust::tuple<idx4, Cell*>> entries(hashtable.get_size());
    for (auto it = zipped; it != zipped + hashtable.get_size(); it++) {
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

// refine the grid by one level
void refineGrid1lvl(map_type& hashtable) {
    size_t numCells = hashtable.get_size();
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));
    // copy to an actual copy of the keys, that won't change as we refine
    thrust::device_vector<thrust::tuple<idx4, Cell*>> entries(hashtable.get_size());
    for (auto it = zipped; it != zipped + hashtable.get_size(); it++) {
        entries[it - zipped] = *it;
    }
    idx4 idx_cell;
    Cell* pCell = nullptr;

    for (auto entry : entries) { // entry is on device
        thrust::tuple<idx4, Cell*> t = entry; // t is on host
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        if (refCrit(pCell->rho) && pCell->flag_leaf) {
            refineGridCell(idx_cell, hashtable);
        }
    }
}

// get information about the neighbor cell necessary for computing the gradient
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_type &hashtable) {
    idx4 idx_neighbor;
    int idx1_parent_neighbor;
    bool is_border, is_notref;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    is_notref = !checkIfExists(idx_neighbor, hashtable) && !is_border;
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
}
__device__ void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_view_type &hashtable) {
    idx4 idx_neighbor;
    int idx1_parent_neighbor;
    bool is_border, is_notref, exists;
    // check if the cell is a border cell
    checkIfBorder(idx_cell, dir, pos, is_border);
    // compute the index of the neighbor on the same level
    getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
    // if the neighbor on the same level does not exist and the cell is not a border cell, then the neighbor is not refined
    checkIfExists(idx_neighbor, hashtable, exists); 
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
__device__ void calcGradCell(const idx4 idx_cell, Cell* cell, map_view_type &hashtable) {
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
__global__ void calcGrad(map_view_type &hashtable, auto zipped, size_t hashtable_size) {
    idx4 idx_cell;
    Cell* pCell = nullptr;
    for (auto it = zipped; it != zipped + hashtable_size; it++) {
        thrust::tuple<idx4, Cell*> t = *it;
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        calcGradCell(idx_cell, pCell, hashtable);
    }
}

void writeGrid(map_type& hashtable) {
    // save i, j, k, L, rho, gradients for all cells (use the iterator) to a file
    ofstream outfile;
    outfile.open(outfile_name);
    idx4 idx_cell;
    Cell* pCell = nullptr;
    outfile << "i,j,k,L,flag_leaf,rho,rho_grad_x,rho_grad_y,rho_grad_z\n";
    size_t numCells = hashtable.get_size();
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));
    for (auto it = zipped; it != zipped + hashtable.get_size(); it++) {
        thrust::tuple<idx4, Cell*> t = *it;
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        outfile << idx_cell.idx3[0] << "," << idx_cell.idx3[1] << "," << idx_cell.idx3[2]
                << "," << idx_cell.L << "," << pCell->flag_leaf << "," << pCell->rho << "," << pCell->rho_grad[0]
                << "," << pCell->rho_grad[1] << "," << pCell->rho_grad[2] << "\n";
    }
    outfile.close();
}

int main() {
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    cout << "Making base grid" << endl;
    makeBaseGrid(grid, hashtable);
    const int num_ref = LMAX - LBASE;
    cout << "Refining grid levels" << endl;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(hashtable);
    }
    cout << "Finished refining grid levels" << endl;
    printHashtableIdx(hashtable);

    cout << "Calculating gradients" << endl;
    auto start = high_resolution_clock::now();

    // run as kernel on GPU
    map_view_type view = hashtable.get_device_view();
    // get zipped values before kicking off kernels
    size_t numCells = hashtable.get_size();
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
    hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));
    calcGrad<<<1, 1>>>(view, zipped, hashtable.get_size());
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    writeGrid(hashtable);
}
