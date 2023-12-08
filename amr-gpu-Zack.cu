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
const int32_t HASH[4] = {-1640531527, 97, 1003313, 5}; // hash function constants

// custom key type
struct idx4 {
    int32_t i, j, k, L;

    __host__ __device__ idx4() {}
    __host__ __device__ idx4(int32_t i_init, int32_t j_init, int32_t k_init, int32_t L_init) : i{i_init}, j{j_init}, k{k_init}, L{L_init} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __device__ bool operator==(idx4 const& other) const {
        return i == other.i && j == other.j && k == other.k && L == other.L;
    }
};

// custom device key equal callable
struct idx4_equals {
    template <typename key_type>
    __device__ bool operator()(key_type const& lhs, key_type const& rhs) {
        return lhs.i == rhs.i && lhs.j == rhs.j && lhs.k == rhs.k && lhs.L == rhs.L;
    }
};

// custom value type
#pragma pack(push, 1)
struct Cell {
    double rho, rho_grad_x, rho_grad_y, rho_grad_z;
    int8_t flag_leaf;

    __host__ __device__ Cell() {}
    __host__ __device__ Cell(double rho_init, double rho_grad_x_init, double rho_grad_y_init, double rho_grad_z_init, int8_t flag_leaf_init) : rho{rho_init}, rho_grad_x{rho_grad_x_init}, rho_grad_y{rho_grad_y_init}, rho_grad_z{rho_grad_z_init}, flag_leaf{flag_leaf_init} {}
};
#pragma pack(pop)

// custom key type hash
struct ramses_hash {
    template <typename key_type>
    __device__ uint32_t operator()(key_type k) {
        int32_t hashval = HASH[0] * k.i + HASH[1] * k.j + HASH[2] * k.k + HASH[3] * k.L;
        return hashval;
    };
};

template<>
struct cuco::is_bitwise_comparable<Cell> : true_type {};

int main() {

    // set empty sentinels
    auto const empty_idx4_sentinel = idx4{-1, -1, -1, -1};
    auto const empty_cell_sentinel = Cell{-1, -1, -1, -1, -1};
    
    // make base grid iterator
    auto pairs_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int32_t>(0),
        [] __device__(auto i) {
            return cuco::make_pair(idx4{i}, Cell{i});
        }
    );

    cuco::static_map<idx4, Cell> hashtable{
        NMAX, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_cell_sentinel}
    };
}