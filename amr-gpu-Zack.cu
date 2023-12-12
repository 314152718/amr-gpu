// defines
#define _USE_MATH_DEFINES

// cpu includes
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

// #include <typeinfo>
// #include <typeindex>

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

// constants
const int32_t LBASE = 3; // base AMR level
const int32_t LMAX = 6; // max AMR level
const int32_t NDIM = 3; // number of dimensions
const int32_t NMAX = 2097152 + 10; // maximum number of cells
const __device__ int32_t HASH[4] = {-1640531527, 97, 1003313, 5}; // hash function constants
const __device__ double FD_KERNEL[4][4] = {
    {-1., 0., 1., 3.},
    {-9., 5., 4., 15.},
    {-4., -5., 9., 15.},
    {-1., 0., 1., 2.}
};
const double rho_crit = 0.01; // critical density for refinement
const double rho_boundary = 0.; // boundary condition
const double sigma = 0.001; // std of Gaussian density field
const double EPS = 0.000001;
const string outfile_name = "grid-gpu.csv";

// custom key type
struct idx4 {
    int32_t idx3[NDIM], L;

    __host__ __device__ idx4() {}
    __host__ __device__ idx4(int32_t i_init, int32_t j_init, int32_t k_init, int32_t L_init) : idx3{i_init, j_init, k_init}, L{L_init} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __device__ bool operator==(idx4 const& other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
};

ostream& operator<<(ostream &os, idx4 const &idx_cell) {
    os << "[" << idx_cell.idx3[0] << ", " << idx_cell.idx3[1] << ", " << idx_cell.idx3[2] << "](L=" << idx_cell.L << ")";
    return os;
}

// custom device key equal callable
struct idx4_equals {
    template <typename key_type>
    __host__ __device__ bool operator()(key_type const& lhs, key_type const& rhs) {
        return lhs.idx3[0] == rhs.idx3[0] && lhs.idx3[1] == rhs.idx3[1] && lhs.idx3[2] == rhs.idx3[2] && lhs.L == rhs.L;
    }
};

// custom value type
struct Cell {
    double rho;
    double rho_grad[3];
    int32_t flag_leaf;

    __host__ __device__ Cell() {}
    __host__ __device__ Cell(double rho_init, double rho_grad_x_init, double rho_grad_y_init, double rho_grad_z_init, 
        int32_t flag_leaf_init) : rho{rho_init}, rho_grad{rho_grad_x_init, rho_grad_y_init, rho_grad_z_init}, flag_leaf{flag_leaf_init} {}

    __host__ __device__ bool operator==(Cell const& other) const {
        return abs(rho - other.rho) < EPS && abs(rho_grad[0] - other.rho_grad[0]) < EPS
            && abs(rho_grad[1] - other.rho_grad[1]) < EPS && abs(rho_grad[2] - other.rho_grad[2]) < EPS;
    }
};

ostream& operator<<(ostream &os, Cell const &cell) {
    os << "[rho " << cell.rho << ", rho_grad_x " << cell.rho_grad[0] << ", rho_grad_y"
       << cell.rho_grad[1] << ", rho_grad_z " << cell.rho_grad[2] << ", flag_leaf " << cell.flag_leaf << "]";
    return os;
}

typedef cuco::static_map<idx4, Cell*> map_type;
typedef cuco::static_map<idx4, Cell*>::device_view map_view_type;

    // custom key type hash
    struct ramses_hash {
    template <typename key_type>
    __host__ __device__ int32_t operator()(key_type k) {
        int32_t hashval = HASH[0] * k.idx3[0] + HASH[1] * k.idx3[1] + HASH[2] * k.idx3[2] + HASH[3] * k.L;
        return hashval;
    };
};

// template<>
// struct cuco::is_bitwise_comparable<Cell> : true_type {};

// --------------- FUNCTION DECLARATIONS ------------ //
void transposeToHilbert(const int X[NDIM], const int L, int &hindex);
void hilbertToTranspose(const int hindex, const int L, int (&X)[NDIM]);
void getHindex(idx4 idx_cell, int& hindex);
void getHindexInv(int hindex, int L, idx4& idx_cell);
void makeBaseGrid(Cell (&grid)[NMAX], map_type &hashtable);
void setGridCell(const idx4 idx_cell, const int hindex, int32_t flag_leaf, map_type &hashtable);
Cell* find(map_type &hashtable, const idx4& key);
void insert(map_type &hashtable, const idx4& key, Cell* const value);
void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_type &hashtable);
void calcGradCell(const idx4 idx_cell, Cell* cell, map_type &hashtable);
void calcGrad(map_type &hashtable);
void writeGrid(map_type &hashtable);
// zip_type retrieve_zipped(map_type &hashtable);

// ------- GLOBALS --------- //


// ------------------------------------------------ //

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

// Compute the Hilbert index for a given 4-idx (i, j, k, L)
void getHindex(idx4 idx_cell, int& hindex) {
    int X[NDIM];
    for (int i=0; i<NDIM; i++){
        X[i] = idx_cell.idx3[i];
    }
    int L = idx_cell.L;
    int m = 1 << (L - 1), p, q, t;
    int i;
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

// Compute the 3-index for a given Hilbert index and AMR level
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

// globals
Cell grid[NMAX];
auto const empty_idx4_sentinel = idx4{-1, -1, -1, -1};
__host__ __device__ Cell* empty_pcell_sentinel = nullptr;

// Multi-variate Gaussian distribution
double rhoFunc(const double coord[NDIM], const double sigma) {
    double rsq = 0;
    for (short i = 0; i < NDIM; i++) {
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
void getParentIdx(const idx4 &idx_cell, idx4 &idx_parent) {
    for (short i = 0; i < NDIM; i++) {
        idx_parent.idx3[i] = idx_cell.idx3[i] / 2;
    }
    idx_parent.L = idx_cell.L - 1;
}

// Compute the indices of the neighbor cells in a given direction
__host__ __device__ void getNeighborIdx(const idx4 &idx_cell, const int dir, const bool pos, idx4 &idx_neighbor) {
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor.idx3[i] = idx_cell.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_cell.L;
}

// void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border) {
//     is_border = idx_cell.idx3[dir] == int(pos) * (pow(2, idx_cell.L) - 1);
// }
__host__ __device__ void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border) {
    is_border = idx_cell.idx3[dir] == int(pos) * (pow(2, idx_cell.L) - 1);
}

Cell* find(map_type& hashtable, const idx4& idx_cell) {
    thrust::device_vector<idx4> key;
    thrust::device_vector<Cell*> value(1);
    key.push_back(idx_cell);
    hashtable.find(key.begin(), key.end(), value.begin());
    // cout << "Searching for " << idx_cell << ", found: " << value[0] << endl;
    return value[0];
}
// GPU version: use map_view_type's find function (just one key at a time)
__device__ void find(map_view_type& hashtable, const idx4& idx_cell, Cell* pCell) {
    cuco::static_map<idx4, Cell*>::device_view::const_iterator pair = hashtable.find(idx_cell);
    // cout << "Searching for " << idx_cell << ", found: " << value[0] << endl;
    pCell = pair->second;
}

// Check if a cell exists
bool checkIfExists(const idx4& idx_cell, map_type &hashtable) {
    Cell* pCell = find(hashtable, idx_cell);
    return pCell != empty_pcell_sentinel;
}
__device__ void checkIfExists(const idx4& idx_cell, map_view_type &hashtable, bool &res) {
    Cell* pCell = nullptr;
    find(hashtable, idx_cell, pCell);
    res = pCell != empty_pcell_sentinel;
}

void makeBaseGrid(Cell (&grid)[NMAX], map_type &hashtable) {
    // not making enough leaves?
    idx4 idx_cell;
    for (int L = 0; L <= LBASE; L++) {
        for (int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            setGridCell(idx_cell, hindex, L == LBASE, hashtable);
        }
    }
};

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

/*
*/
// TODO: this could probably also run on GPU (using a device view)
void insert(map_type &hashtable, const idx4& key, Cell* const value) {
    thrust::device_vector<idx4> insert_keys;
    thrust::device_vector<Cell*> insert_values;
    insert_keys.push_back(key);
    insert_values.push_back(value);
    // todo change this to just inserting using a pair, zip is unnecessary
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

    hashtable.insert(zipped, zipped + insert_keys.size());
}

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


void refineGridCell(const idx4 idx_cell, map_type &hashtable) {
    int hindex;
    getHindex(idx_cell, hindex);

    Cell *pCell = find(hashtable, idx_cell);
    if (pCell == empty_pcell_sentinel) throw runtime_error("Trying to refine non-existant cell!");

    if (!pCell->flag_leaf) throw runtime_error("trying to refine non-leaf");
    if (idx_cell.L == LMAX) throw runtime_error("trying to refine at max level");
    
    // make this cell a non-leaf
    pCell->flag_leaf = 0;

    idx4 idx_child = idx_cell;
    idx_child.L++;
    for (short dir = 0; dir < NDIM; dir++) idx_child.idx3[dir] *= 2;

    // and create 2^NDIM leaf children
    setChildrenHelper(idx_child, 0, hashtable);

    // refine neighbors if needed
    idx4 idx_neighbor, idx_parent;
    int hindex_neighbor;
    for (short dir = 0; dir < NDIM; dir++) {
        for (short pos = 0; pos < 2; pos++) {
            bool is_border;
            checkIfBorder(idx_cell, dir, pos, is_border);
            if (is_border) continue;
            getNeighborIdx(idx_cell, dir, pos, idx_neighbor);
            // don't need to remove 'if' statements because this is part not for GPU (only gradient is)
            // don't need to refine if exists
            if (checkIfExists(idx_neighbor, hashtable)) continue;

            // if not exists, drop L by differen
            // we assume that L is at most different by 1
            getParentIdx(idx_cell, idx_parent);
            getNeighborIdx(idx_parent, dir, pos, idx_neighbor);
            refineGridCell(idx_neighbor, hashtable);
        }
    }
}

// zip_type retrieve_zipped(map_type& hashtable) {
//     size_t numCells = hashtable.get_size();
//     thrust::device_vector<idx4> retrieved_keys(numCells);
//     thrust::device_vector<Cell*> retrieved_values(numCells);
//     hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin());               // doesn't populate values for some reason
//     hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values
//     zip_type zipped = 
//         thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));

//     // // Use typeid and type_index to get type information
//     // const std::type_info& typeInfo = typeid(zipped);
//     // std::type_index typeIndex = std::type_index(typeInfo);
//     // cout << "zip type: " << typeIndex.name() << endl;

//     return zipped;
// }

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
            refineGridCell(idx_cell, hashtable); // refinement step is failing
        }
    }
}

// get information about the neighbor cell necessary for computing the gradient
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
// GPU VERISON: get information about the neighbor cell necessary for computing the gradient
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



// all tests (later move out)

void test_full_output() {
    // Cell empty_cell_sentinel{-1, -1, -1, -1, -1};
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    // grid memory accessible from CPU or GPU?
    // cudaMallocManaged(&x, N * sizeof(float));

    cout << "Making base grid" << endl;
    makeBaseGrid(grid, hashtable);
    const int num_ref = LMAX - LBASE;
    cout << "Refining grid levels" << endl;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(hashtable);
    }
    cout << "Finished refining grid levels" << endl;

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
    calcGrad<<<8, 2>>>(view, zipped, hashtable.get_size());

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << duration.count() << " ms" << endl;
    writeGrid(hashtable);
}

void test_map_insert_int() {
    cuco::static_map<int32_t, int32_t> hashtable{NMAX, cuco::empty_key{-1}, cuco::empty_value{-1}};
    thrust::device_vector<cuco::pair<int32_t, int32_t>> test_pair;
    test_pair.push_back(pair<int32_t, int32_t>(2, 3));
    hashtable.insert(test_pair.begin(), test_pair.end()); // was not working

    // Retrieve contents of all the non-empty slots in the map
    thrust::device_vector<int32_t> result_keys(2);
    thrust::device_vector<int32_t> result_values(2);
    hashtable.retrieve_all(result_keys.begin(), result_values.begin());

    cout << "KEYS:" << endl;
    for (auto k : result_keys) {
        cout << k << endl;
    }

    cout << "VALUES:" << endl;
    for (auto v : result_values) {
        cout << v << endl;
    }
}

void test_map_insert_cell_pointer() {
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    // Retrieve contents of all the non-empty slots in the map
    thrust::device_vector<idx4> result_keys(2);
    thrust::device_vector<Cell*> result_values(2);
    // hashtable.find(insert_keys.begin(), insert_keys.end(), result_values.begin());
    // hashtable.retrieve_all(result_keys.begin(), result_values.begin());

    // cout << "KEYS:" << endl;
    // for (auto k : result_keys) {
    //     cout << k << endl;
    // }

    Cell* pResult;
    cout << "VALUES:" << endl;
    for (auto v : result_values) {
        // v is a pointer to Cell
        pResult = v;
        cout << pResult << endl;
    }

    //delete pTest_cell;
}
 
void test_map_insert_cell_pointer_Roma() {
    idx4 idx_cell{1, 1, 1, 1};
    Cell* pTest_cell = new Cell{1, 1, 1, 1, 1}; // create on heap
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    cout << "address of test_cell:" << pTest_cell << endl;
    cout << "test cell rho:" << pTest_cell->rho << endl;

    thrust::device_vector<idx4> insert_keys;
    insert_keys.push_back(idx_cell);
    thrust::device_vector<Cell*> insert_values;
    insert_values.push_back(pTest_cell);
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

    // trying zip iterator
    hashtable.insert(zipped, zipped + insert_keys.size());

    bool test_exist;
    test_exist = checkIfExists(idx_cell, hashtable);
    cout << "KEY EXISTS? " << test_exist << endl;
    test_exist = checkIfExists(idx4{1,2,3,4}, hashtable);
    cout << "FAKE KEY EXISTS? " << test_exist << endl;

    // trying retrieve all
    // Retrieve contents of all the non-empty slots in the map
    // why retrieve all not working?
    thrust::device_vector<idx4> result_keys(2);
    thrust::device_vector<Cell*> result_values(2);
    hashtable.retrieve_all(result_keys.begin(), result_values.begin());
    hashtable.find(result_keys.begin(), result_keys.end(), result_values.begin());

    cout << "KEYS:" << endl;
    for (auto k : result_keys) {
        cout << k << endl;
    }

    cout << "VALUES:" << endl;
    for (auto v : result_values) {
        cout << v << endl;
    }
}

void test_map_insert_cell_pointer_Roma2() {
    using Key = idx4;
    using Value = Cell*;
    // cuco::static_map<Key, Value> hashtable{NMAX, cuco::empty_key{-1}, cuco::empty_value{-1}};

    idx4 idx_cell{1, 1, 1, 1};
    Cell* pTest_cell = new Cell{1, 1, 1, 1, 1}; // create on heap
    cout << "Address of test cell: " << pTest_cell << endl;
    cuco::static_map<Key, Value> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    thrust::device_vector<cuco::pair<Key, Value>> test_pair;
    // test_pair.push_back(pair<Key, Value>(2, 3));
    test_pair.push_back(pair<Key, Value>(idx_cell, pTest_cell));
    hashtable.insert(test_pair.begin(), test_pair.end());

    // check if its in the table correctly?
    Cell* pResult;
    pResult = find(hashtable, idx_cell);
    cout << "Found" << pResult << endl;
    cout << "Size: " << hashtable.get_size() << endl;

    // Retrieve contents of all the non-empty slots in the map
    thrust::device_vector<Key> result_keys(1);
    thrust::device_vector<Value> result_values(1);
    // roundabout solution since retrieve_all not getting values: use it to get all keys, and then pass into find to get all values
    hashtable.retrieve_all(result_keys.begin(), result_values.begin());
    hashtable.find(result_keys.begin(), result_keys.end(), result_values.begin());

    cout << "KEYS:" << endl;
    for (auto k : result_keys) {
        cout << k << endl;
    }

    cout << "VALUES:" << endl;
    for (auto v : result_values) {
        cout << v << endl;
    }
}

void testHilbert() {
    idx4 idx_cell, idx_cell2;
    cin >> idx_cell.idx3[0];
    cin >> idx_cell.idx3[1];
    cin >> idx_cell.idx3[2];
    idx_cell.L = 2;
    int hindex;
    getHindex(idx_cell, hindex);
    cout << hindex << endl;
    // test inverse
    getHindexInv(hindex, 2, idx_cell2);
    cout << "Inverse of hindex=" << hindex << " is " << idx_cell2 << endl;
}

int main() {
    test_full_output();
}
