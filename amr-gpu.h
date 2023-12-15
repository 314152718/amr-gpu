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

// --------------- STRUCTS ------------ //
// custom key type
struct idx4 {
    int32_t idx3[NDIM], L;

    __host__ __device__ idx4() {}
    __host__ __device__ idx4(int32_t i_init, int32_t j_init, int32_t k_init, int32_t L_init) : idx3{i_init, j_init, k_init}, L{L_init} {}
    __host__ __device__ idx4(const int32_t ijk_init[NDIM], int32_t L_init) : idx3{ijk_init[0], ijk_init[1], ijk_init[2]}, L{L_init} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __device__ bool operator==(idx4 const& other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
    // __device__: identifier std basic string is undefined in device code
    string str() const {
        return "["+to_string(idx3[0])+", "+to_string(idx3[1])+", "+to_string(idx3[2])+"](L="+to_string(L)+")";
    }
};

// custom key output stream representation
ostream& operator<<(ostream &os, idx4 const &idx_cell) {
    os << "[" << idx_cell.idx3[0] << ", " << idx_cell.idx3[1] << ", " << idx_cell.idx3[2] << "](L=" << idx_cell.L << ")";
    return os;
}

// custom key equal callable
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

// custom value output stream representation
ostream& operator<<(ostream &os, Cell const &cell) {
    os << "[rho " << cell.rho << ", rho_grad_x " << cell.rho_grad[0] << ", rho_grad_y"
       << cell.rho_grad[1] << ", rho_grad_z " << cell.rho_grad[2] << ", flag_leaf " << cell.flag_leaf << "]";
    return os;
}
// ------------------------------------------------ //

// typedefs
typedef cuco::static_map<idx4, Cell*> map_type;
typedef cuco::static_map<idx4, Cell*>::device_view map_view_type;

// globals
Cell grid[NMAX];
auto const empty_idx4_sentinel = idx4{-1, -1, -1, -1};
__host__ __device__ Cell* empty_pcell_sentinel = nullptr;

// --------------- FUNCTION DECLARATIONS ------------ //
void transposeToHilbert(const int X[NDIM], const int L, int &hindex);
void hilbertToTranspose(const int hindex, const int L, int (&X)[NDIM]);
void getHindex(idx4 idx_cell, int& hindex);
void getHindexInv(int hindex, int L, idx4& idx_cell);
double rhoFunc(const double coord[NDIM], const double sigma);
bool refCrit(double rho);
void getParentIdx(const idx4 &idx_cell, idx4 &idx_parent);
__host__ __device__ void getNeighborIdx(const idx4 idx_cell, const int dir, const bool pos, idx4 idx_neighbor);
__host__ __device__ void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border);
Cell* find(map_type& hashtable, const idx4& idx_cell);
__device__ void find(map_view_type &hashtable, const idx4 idx_cell, Cell *pCell);
bool checkIfExists(const idx4& idx_cell, map_type &hashtable);
__device__ void checkIfExists(const idx4 idx_cell, map_view_type &hashtable, bool &res);
void makeBaseGrid(Cell (&grid)[NMAX], map_type &hashtable);
void setGridCell(const idx4 idx_cell, const int hindex, int32_t flag_leaf, map_type &hashtable);
void insert(map_type &hashtable, const idx4& key, Cell* const value);
void setChildrenHelper(idx4 idx_cell, short i, map_type &hashtable);
void refineGridCell(const idx4 idx_cell, map_type &hashtable);
void printHashtableIdx(map_type& hashtable);
void refineGrid1lvl(map_type& hashtable);
void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_type &hashtable);
__device__ void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor, map_view_type &hashtable);
__device__ void calcGradCell(const idx4 idx_cell, Cell* cell, map_view_type &hashtable);
__global__ void calcGrad(map_view_type &hashtable, auto zipped, size_t hashtable_size);
void writeGrid(map_type& hashtable);
// ------------------------------------------------ //