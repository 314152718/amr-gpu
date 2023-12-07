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
const int32_t L_base = 3; // base AMR level
const int32_t L_max = 6; // max AMR level
const int32_t N_dim = 3; // number of dimensions
const int32_t N_cell_max = 2097152 + 10; // maximum number of cells
const int32_t hash_constants[4] = {-1640531527, 97, 1003313, 5}; // hash function constants

// custom key type
#pragma pack(push, 1)
struct idx4 {
    int32_t i;
    int32_t j;
    int32_t k;
    int32_t L;

    __host__ __device__ idx4() {}
    __host__ __device__ idx4(int32_t i_init, int32_t j_init, int32_t k_init, int32_t L_init) : i{i_init}, j{j_init}, k{k_init}, L{L_init} {}

    // Device equality operator is mandatory due to libcudacxx bug:
    // https://github.com/NVIDIA/libcudacxx/issues/223
    __device__ bool operator==(idx4 const& other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
};
#pragma pack(pop)

// custom device key equal callable
struct idx4_equals {
    template <typename key_type>
    __device__ bool operator()(key_name const& lhs, key_type const& rhs) {
        return lhs.idx3[0] == rhs.idx3[0] && lhs.idx3[1] == rhs.idx3[1] && lhs.idx3[2] == rhs.idx3[2] && lhs.idx3[3] == rhs.idx3[3];
    }
};

// custom value type
#pragma pack(push, 1)
struct Cell {
    double rho, rho_grad_x, rho_grad_y, rho_grad_z;

    __host__ __device__ Cell() {}
    __host__ __device__ Cell(double rho_init, double rho_grad_x_init, double rho_grad_y_init, double rho_grad_z_init, int8_t flag_leaf_init) : rho{rho_init}, rho_grad_x{rho_grad_x_init}, rho_grad_y{rho_grad_y_init}, rho_grad_z{rho_grad_z_init}, flag_leaf{flag_leaf_init} {}
};
#pragma pack(pop)

// custom key type hash
struct ramses_hash {
    template <typename key_type>
    __device__ uint32_t operator()(key_type k) {
        uint8_t hashval = 0;
        for (uint8_t i = 0; i < N_dim; i++) {
            hashval += idx_cell.idx3[i] * hash_constants[i];
        }
        hashval += idx_cell.L * hash_constants[3];
        return hashval;
    };
};

template<>
struct cuco::is_bitwise_comparable<Cell> : true_type {};

int main() {

    // set empty sentinels
    auto const empty_idx4_sentinel = idx4{-1};
    auto const empty_cell_sentinel = Cell{-1};
    
    // create an interator of input key/value pairs
    auto pairs_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator<int32_t>(0),
        [] __device__(auto i) {
            return cuco::make_pair(idx4{i}, Cell{i});
        }
    );

    cuco::static_map<idx4, Cell> hashtable{
        N_cell_max, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_cell_sentinel};
    }
}