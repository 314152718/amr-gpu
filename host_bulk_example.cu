/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <limits>
// --------- CUSTOM TYPES ---------- //

// constants
const int32_t LBASE = 3; // base AMR level
const int32_t LMAX = 6; // max AMR level
const int32_t NDIM = 3; // number of dimensions
const int32_t NMAX = 2097152 + 10; // maximum number of cells
const __device__ int32_t HASH[4] = {-1640531527, 97, 1003313, 5}; // hash function constants
const double rho_crit = 0.01; // critical density for refinement
const double sigma = 0.001; // std of Gaussian density field
const double EPS = 0.000001;

#pragma pack(1)
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
std::ostream& operator<<(std::ostream &os, const idx4 &idx) {
    os << "[" << idx.idx3[0] << ", " << idx.idx3[1] << ", " << idx.idx3[2] << "](L=" << idx.L << ")";
    return os;
}

// custom device key equal callable
struct idx4_equals {
    template <typename key_type>
    __device__ bool operator()(key_type const& lhs, key_type const& rhs) {
        return lhs.idx3[0] == rhs.idx3[0] && lhs.idx3[1] == rhs.idx3[1] && lhs.idx3[2] == rhs.idx3[2] && lhs.L == rhs.L;
    }
};

// template<>
// struct cuco::is_bitwise_comparable<Cell> : true_type {};

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

typedef cuco::static_map<idx4, Cell*> map_type;

// custom key type hash
struct ramses_hash {
    template <typename key_type>
    __device__ uint32_t operator()(key_type k) {
        int32_t hashval = HASH[0] * k.idx3[0] + HASH[1] * k.idx3[1] + HASH[2] * k.idx3[2] + HASH[3] * k.L;
        return hashval;
    };
};

auto const empty_idx4_sentinel = idx4{-1, -1, -1, -1};
Cell* empty_pcell_sentinel = nullptr;


/**
 * @file host_bulk_example.cu
 * @brief Demonstrates usage of the static_map "bulk" host APIs.
 *
 * The bulk APIs are only invocable from the host and are used for doing operations like insert or
 * find on a set of keys.
 *
 */

int main(void) {
    using Key = idx4;
    using Value = Cell*;

    // test values
    idx4 idx_cell{1, 1, 1, 1};
    Cell* pTest_cell = new Cell{1, 1, 1, 1, 1}; // create on heap

    // Empty slots are represented by reserved "sentinel" values. These values should be selected such
    // that they never occur in your input data.
    // Key constexpr empty_key_sentinel = -1;
    // Value constexpr empty_value_sentinel = -1;

    // Number of key/value pairs to be inserted
    std::size_t constexpr num_keys = 1;
    std::size_t const capacity = 10;

    // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
    cuco::static_map<Key, Value> map{
        capacity, cuco::empty_key{empty_idx4_sentinel}, cuco::empty_value{empty_pcell_sentinel}};

    // Create a sequence of keys and values {{0,0}, {1,1}, ... {i,i}}
    thrust::device_vector<Key> insert_keys;
    insert_keys.push_back(idx_cell);
    thrust::device_vector<Value> insert_values;
    insert_values.push_back(pTest_cell);
    auto zipped =
        thrust::make_zip_iterator(thrust::make_tuple(insert_keys.begin(), insert_values.begin()));

    // Inserts all pairs into the map
    map.insert(zipped, zipped + insert_keys.size());

    // Storage for found values
    thrust::device_vector<Value> found_values(num_keys);

    // Finds all keys {0, 1, 2, ...} and stores associated values into `found_values`
    // If a key `keys_to_find[i]` doesn't exist, `found_values[i] == empty_value_sentinel`
    map.find(insert_keys.begin(), insert_keys.end(), found_values.begin());

    Cell* pCell;
    std::cout << " FOUND VALUES " << std::endl;
    for (auto v : found_values) {
      pCell = v;
      std::cout << pCell << std::endl;
    }

    // Verify that all the found values match the inserted values
    bool const all_values_match =
        thrust::equal(found_values.begin(), found_values.end(), insert_values.begin());

    if (all_values_match) {
        std::cout << "Success! Found all values.\n";
    }

    return 0;
}