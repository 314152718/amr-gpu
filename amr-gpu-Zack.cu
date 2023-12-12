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
const double rho_crit = 0.01; // critical density for refinement
const double sigma = 0.001; // std of Gaussian density field
const double EPS = 0.000001;

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
    int32_t rho, rho_grad_x, rho_grad_y, rho_grad_z;
    int8_t flag_leaf;

    __host__ __device__ Cell() {}
    __host__ __device__ Cell(int32_t rho_init, int32_t rho_grad_x_init, int32_t rho_grad_y_init, int32_t rho_grad_z_init, 
        int8_t flag_leaf_init) : rho{rho_init}, rho_grad_x{rho_grad_x_init}, rho_grad_y{rho_grad_y_init}, 
        rho_grad_z{rho_grad_z_init}, flag_leaf{flag_leaf_init} {}

    __host__ __device__ bool operator==(Cell const& other) const {
        return abs(rho - other.rho) < EPS && abs(rho_grad_x - other.rho_grad_x) < EPS
            && abs(rho_grad_y - other.rho_grad_y) < EPS && abs(rho_grad_z - other.rho_grad_z) < EPS;
    }
};

ostream& operator<<(ostream &os, Cell const &cell) {
    os << "[rho " << cell.rho << ", rho_grad_x " << cell.rho_grad_x << ", rho_grad_y"
       << cell.rho_grad_y << ", rho_grad_z " << cell.rho_grad_z << ", flag_leaf " << cell.flag_leaf << "]";
    return os;
}

typedef cuco::static_map<idx4, Cell*> map_type;

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
void setGridCell(const idx4 idx_cell, const int hindex, int8_t flag_leaf, map_type &hashtable);
Cell* find(map_type &hashtable, const idx4& key);
void insert(map_type &hashtable, const idx4& key, Cell* const value);

// ------- GLOBALS --------- //

thrust::device_vector<idx4> insert_keys;
thrust::device_vector<Cell*> insert_values;


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
        X[0] = idx_cell.idx3[i];
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
Cell* empty_pcell_sentinel = nullptr;

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
void getNeighborIdx(const idx4 &idx_cell, const int dir, const bool pos, idx4 &idx_neighbor) {
    for (short i = 0; i < NDIM; i++) {
        idx_neighbor.idx3[i] = idx_cell.idx3[i] + (int(pos) * 2 - 1) * int(i == dir);
    }
    idx_neighbor.L = idx_cell.L;
}

void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border) {
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

// Check if a cell exists
bool checkIfExists(const idx4& idx_cell, map_type &hashtable) {
    Cell* pCell = find(hashtable, idx_cell);
    // cout <<  "pCell != empty_pcell_sentinel: " << (pCell != empty_pcell_sentinel) << endl;
    return pCell != empty_pcell_sentinel;
}

void makeBaseGrid(Cell (&grid)[NMAX], map_type &hashtable) {
    idx4 idx_cell;
    for (int L = 0; L <= LBASE; L++) {
        for (int hindex = 0; hindex < pow(2, NDIM * L); hindex++) {
            getHindexInv(hindex, L, idx_cell);
            setGridCell(idx_cell, hindex, L == LBASE, hashtable);
        }
    }
};

void setGridCell(const idx4 idx_cell, const int hindex, int8_t flag_leaf, map_type &hashtable) {
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

    // INSERT INTO HASHTABLE
    // hashtable[idx_cell] = &grid[offset + hindex];
    insert(hashtable, idx_cell, &grid[offset + hindex]);
}

/*
*/
// TODO: this could probably also run on GPU (using a device view)
void insert(map_type &hashtable, const idx4& key, Cell* const value) {
    if (insert_keys.size() < 1) {
        insert_keys.push_back(key);
    } else {
        insert_keys[0] = key;
    }
    if (insert_values.size() < 1) {
        insert_values.push_back(value);
    } else {
        insert_values[0] = value;
    }
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

    Cell *pCell;
    pCell = find(idx_cell, hashtable);
    if (pCell == empty_pcell_sentinel) throw runtime_error("Trying to refine non-existant cell!");

    if (!pCell->flag_leaf) throw runtime_error("trying to refine non-leaf");
    if (idx_cell.L == LMAX) throw runtime_error("trying to refine at max level");
    pCell->flag_leaf = 0;

    idx4 idx_child = idx_cell;
    idx_child.L++;
    for (short dir = 0; dir < NDIM; dir++) idx_child.idx3[dir] *= 2;

    // todo: fix bug where it doesn't actually go thru all the permutations
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

void refineGrid1lvl(map_type& hashtable) {
    // rewrite with retrieve_all
    size_t numCells = hashtable.get_size();
    thrust::device_vector<idx4> retrieved_keys(numCells);
    thrust::device_vector<Cell*> retrieved_values(numCells);
    hashtable.retrieve_all(retrieved_keys.begin(), retrieved_values.begin()); // doesn't populate values for some reason
    hashtable.find(retrieved_keys.begin(), retrieved_keys.end(), retrieved_values.begin()); // this will populate values

    auto zipped = thrust::make_zip_iterator(thrust::make_tuple(retrieved_keys.begin(), retrieved_values.begin()));
    idx4 idx_cell;
    Cell* pCell;
    for (auto it = zipped; it != zipped + retrieved_keys.size(); it++) {
        thrust::tuple<idx4, Cell*> t = *it;
        idx_cell = t.get<0>();
        pCell = t.get<1>();
        cout << "Retrieved pair: " << idx_cell << ", " << pCell << endl;
        if (refCrit(pCell->rho) && pCell->flag_leaf) {
            refineGridCell(idx_cell, hashtable);
        }
    }
}


// all tests (later move out)

void test_hashtable_set_leafflag() {
    cout << "HELLO" << endl;
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

    makeBaseGrid(grid, hashtable);
    const int num_ref = LMAX - LBASE;
    for (short i = 0; i < num_ref; i++) {
       refineGrid1lvl(hashtable);
    }

    // auto start = high_resolution_clock::now();
    // calcGrad();
    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);
    // cout << duration.count() << " ms" << endl;
    // writeGrid();

    thrust::device_vector<idx4> result_keys(73);
    thrust::device_vector<Cell*> result_values(73);
    hashtable.find(insert_keys.begin(), insert_keys.end(), result_values.begin());

    cout << "KEYS:" << endl;
    for (auto k : result_keys) {
        cout << k << endl;
    }

    cout << "VALUES:" << endl;
    for (auto v : result_values) {
        cout << *v << endl;
    }
    cout << endl << endl;

    idx4 idx_cell{0, 0, 0, 1};
    cout << find(hashtable, idx_cell)->flag_leaf << endl;
    find(hashtable, idx_cell)->flag_leaf = 1;
    cout << find(hashtable, idx_cell)->flag_leaf << endl;
}

int main() {
    test_hashtable_set_leafflag();
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
    hashtable.find(insert_keys.begin(), insert_keys.end(), result_values.begin());
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
    cuco::static_map<idx4, Cell*> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}
    };

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
    pResult = find(idx_cell, hashtable);
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