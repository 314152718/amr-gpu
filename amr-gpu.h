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
#include <limits>

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

// constants
const int32_t LBASE = 2; // 3; base AMR level
const int32_t LMAX = 9; // max AMR level

const int32_t NDIM = 3; // number of dimensions
// expression must have a constant value

// uint32_t max val 4 294 967 295
// lround(2*pow(2, LMAX*NDIM)); <- dont work
constexpr uint32_t NCELL_MAX_ARR[11] = {1, 9, 73, 585, 4681, 37449, 299593, 2396745, 19173961, 153391689, 1227133513};
constexpr uint32_t NCELL_MAX = 1227133513; //2396745 (7); //1227133513 (10); // 2097152 + 10;
//2*2^15 = 70368744177664; 

const int32_t IDX_MAX = pow(2, LMAX) - 1;
const __device__ double FD_KERNEL[4][4] = {
    {-1., 0., 1., 3.},
    {-9., 5., 4., 15.},
    {-4., -5., 9., 15.},
    {-1., 0., 1., 2.}
};
const double rho_crit = 0.01; // critical density for refinement
const double rho_boundary = 0.; // boundary condition
const double sigma = 0.01; // std of Gaussian density field
const double EPS = 1E-12; // 1e-6 is too large for level 7 in getHindexAndOffset
const double STEP_EPS = 0.00001;

// GPU consts
auto constexpr BLOCK_SIZE = 256; //32*n
auto const GRID_SIZE  = (NCELL_MAX + BLOCK_SIZE - 1) / BLOCK_SIZE;

typedef unsigned short int uint16;
auto const uint16_nan = numeric_limits<uint16>::quiet_NaN(); // size_t = uint16
auto const double_nan = numeric_limits<double>::quiet_NaN();
auto const int32t_nan = numeric_limits<int32_t>::quiet_NaN();

// --------------- STRUCTS ------------ //
// custom key type
struct idx4 {
    uint16 idx3[NDIM]; 
    short int L;

    __host__ __device__ idx4() = default;
    __host__ __device__ idx4(uint16 i_init, uint16 j_init, uint16 k_init, short int L_init) : idx3{i_init, j_init, k_init}, L{L_init} {}
    __host__ __device__ idx4(const uint16 ijk_init[NDIM], short int L_init) : idx3{ijk_init[0], ijk_init[1], ijk_init[2]}, L{L_init} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __host__ __device__ bool operator==(idx4 const& other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
    // __device__: identifier std basic string is undefined in device code
    __host__ __device__ void print() const {
        printf("[%d, %d, %d](L=%d)", idx3[0], idx3[1], idx3[2], L);
    }
    __host__ __device__ void println() const {
        printf("[%d, %d, %d](L=%d)\n", idx3[0], idx3[1], idx3[2], L);
    }
    __host__ string str() const {
        return "["+to_string(idx3[0])+", "+to_string(idx3[1])+", "+to_string(idx3[2])+"](L="+to_string(L)+")";
    }

    /* // bitwise comparable errors from this:
    __host__ __device__ idx4 (const idx4 &other) { 
        L = other.L;
        for (short i = 0; i < NDIM; i++) {
            idx3[i] = other.idx3[i];
        }
    }*/
};

namespace std
{
    template<>
    struct hash<idx4> {
        // size_t ?? == uint16 -> max 65000
        __host__ __device__ size_t operator()(const idx4& idx_cell) const noexcept {
            // does not want to define it on device if put outside this function
            const int32_t HASH[4] = {-1640531527, 97, 1003313, 5};
            size_t hashval = HASH[0] * idx_cell.idx3[0] + HASH[1] * idx_cell.idx3[1] + HASH[2] * idx_cell.idx3[2] + 
                HASH[3] * idx_cell.L;
            return hashval;
        };
    };
}

// custom key output stream representation
ostream& operator<<(ostream &os, idx4 const &idx_cell) {
    os << "[" << idx_cell.idx3[0] << ", " << idx_cell.idx3[1] << ", " << idx_cell.idx3[2] << "](L=" << idx_cell.L << ")";
    return os;
}

// custom key equals callable

struct idx4_equals {
    template <typename key_type>
    __host__ __device__ bool operator()(key_type const& lhs, key_type const& rhs) {
        return lhs.idx3[0] == rhs.idx3[0] && lhs.idx3[1] == rhs.idx3[1] && lhs.idx3[2] == rhs.idx3[2] && lhs.L == rhs.L;
    }
};

// custom key type hash
struct ramses_hash {
    template <typename key_type>
    __host__ __device__ uint32_t operator()(key_type k) {
        const int32_t HASH[4] = {-1640531527, 97, 1003313, 5};
        int32_t hashval = HASH[0] * k.idx3[0] + HASH[1] * k.idx3[1] + HASH[2] * k.idx3[2] + HASH[3] * k.L;
        return hashval;
    };
};

// custom value type
struct Cell {
    float rho;
    // explicitly use NDIM == 3
    float rho_grad[3];
    int32_t flag_leaf;

    __host__ __device__ Cell() {
        rho = 0.0;
        for (int i = 0; i < NDIM; i++) 
            rho_grad[i] = 0.0;
        flag_leaf = 0;
    }
    /*__host__ __device__ Cell(Cell &other) {
        rho = other.rho;
        for (int i = 0; i < NDIM; i++) 
            rho_grad[i] = other.rho_grad[i];
        flag_leaf = other.flag_leaf;
    }*/
    __host__ __device__ Cell(float rho_init, float rho_grad_x_init, float rho_grad_y_init, float rho_grad_z_init, 
        int32_t flag_leaf_init) : rho{rho_init}, rho_grad{rho_grad_x_init, rho_grad_y_init, rho_grad_z_init}, flag_leaf{flag_leaf_init} {}

    __host__ __device__ bool operator==(Cell const& other) const {
        return abs(rho - other.rho) < EPS && abs(rho_grad[0] - other.rho_grad[0]) < EPS
            && abs(rho_grad[1] - other.rho_grad[1]) < EPS && abs(rho_grad[2] - other.rho_grad[2]) < EPS;
    }
    __host__ __device__ void print() const {
        printf("[%.2f, %.2f, %.2f, %.2f](Leaf=%d)", rho, rho_grad[0], rho_grad[1], rho_grad[2], flag_leaf);
    }
    __host__ __device__ void println() const {
        print();
        printf("\n");
    }
    __host__ string str() const {
        return "["+to_string(rho)+", "+to_string(rho_grad[0])+", "+to_string(rho_grad[1])+", "+to_string(rho_grad[2])+
                "](Leaf="+to_string(flag_leaf)+")";
    }
};

__host__ __device__ void println(Cell const &cell) {
    cell.println();
}

// on host
// purpose is to store size as well
//typedef cuco::static_map<idx4, Cell*, cuco::extent<std::size_t, 2097162UL>, cuda::std::__4::__detail::thread_scope_device, idx4_equals, cuco::linear_probing<1, ramses_hash>, cuco::cuda_allocator<cuco::pair<idx4, Cell *>>

// custom value output stream representation
ostream& operator<<(ostream &os, Cell const &cell) {
    os << "[rho " << cell.rho << ", rho_grad_x " << cell.rho_grad[0] << ", rho_grad_y"
       << cell.rho_grad[1] << ", rho_grad_z " << cell.rho_grad[2] << ", flag_leaf " << cell.flag_leaf << "]";
    return os;
}
// ------------------------------------------------ //

// typedefs
typedef cuco::static_map<idx4, Cell*> map_type;
typedef unordered_map<idx4, Cell> host_map; //, ramses_hash<idx4>, idx4_equals<idx4>
//typedef cuco::static_map<idx4, Cell*>::device_view map_view_type;

// globals
auto const empty_idx4_sentinel = idx4{0, 0, 0, -1}; // works same as with uint16_nan: cannot use 0, 0, 0, -1 idx4
auto const empty_cell_sentinel = Cell{double_nan, double_nan, double_nan, double_nan, int32t_nan};
Cell* empty_pcell_sentinel = nullptr;

// --------------- FUNCTION DECLARATIONS ------------ //
void checkLast(const char* const file, const int line);
// convert from transposed Hilbert index to Hilbert index
__host__ __device__ void transposeToHilbert(const int X[NDIM], const short int L, long int &hindex) {
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
__host__ __device__ void hilbertToTranspose(const long int hindex, const int L, int *X) {
    long int h = hindex;
    for (short i = 0; i < NDIM; ++i) X[i] = 0;
    for (short i = 0; i < NDIM * L; ++i) {
        short a = (NDIM - (i % NDIM) - 1);
        X[a] |= (h & 1) << (i / NDIM);
        h >>= 1;
    }
}
