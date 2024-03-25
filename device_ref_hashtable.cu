/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

/*EXAMPLE POINTER CAST

thrust::device_vector<int> dv(10);
int * dv_ptr = thrust::raw_pointer_cast(dv.data());
kernel<<<bl,tpb>>>(dv_ptr)

*/
 
 #define _USE_MATH_DEFINES

 #include <iostream>
 #include <stdio.h>
 #include <fstream>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <unordered_map>
 #include <stdexcept>
 #include <chrono>
 
 #include <cuco/static_map.cuh>

 #include <thrust/device_vector.h>
 #include <thrust/execution_policy.h>
 #include <thrust/iterator/zip_iterator.h>
 #include <thrust/logical.h>
 #include <thrust/sequence.h>
 #include <thrust/tuple.h>
 
 #include <cstddef>
 #include <limits>

 using namespace std;
 using namespace std::chrono;

 const int32_t NDIM = 3; // number of dimensions
 const int32_t NCELL_MAX = 2097152 + 10; // maximum number of cells
 const double EPS = 0.000001;
 const __device__ int32_t HASH[4] = {-1640531527, 97, 1003313, 5}; // hash function constants
 typedef unsigned short int uint16;
 const uint16 uint16NaN = numeric_limits<uint16>::quiet_NaN(); 
 const double doubleNaN = numeric_limits<double>::quiet_NaN(); 
 const int32_t int32_tNaN = numeric_limits<int32_t>::quiet_NaN(); 

 struct idx4 {
  uint16 idx3[NDIM], L;

  __host__ __device__ idx4() {}
  __host__ __device__ idx4(uint16 i_init, uint16 j_init, uint16 k_init, uint16 L_init) : idx3{i_init, j_init, k_init}, L{L_init} {}
  __host__ __device__ idx4(const uint16 ijk_init[NDIM], uint16 L_init) : idx3{ijk_init[0], ijk_init[1], ijk_init[2]}, L{L_init} {}

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
  __device__ uint32_t operator()(key_type k) {
      int32_t hashval = HASH[0] * k.idx3[0] + HASH[1] * k.idx3[1] + HASH[2] * k.idx3[2] + HASH[3] * k.L;
      return hashval;
  };
};

// custom value type
__device__ struct Cell {
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

 
 /**
  * @file device_ref_example.cu
  * @brief Demonstrates usage of the device side APIs for individual operations like insert/find.
  *
  * Individual operations like a single insert or find can be performed in device code via the
  * "static_map_ref" types.
  *
  * @note This example is for demonstration purposes only. It is not intended to show the most
  * performant way to do the example algorithm.
  *
  */
 
 /**
  * @brief Inserts keys that pass the specified predicated into the map.
  *
  * @tparam Map Type of the map device reference
  * @tparam KeyIter Input iterator whose value_type convertible to Map::key_type
  * @tparam ValueIter Input iterator whose value_type is convertible to Map::mapped_type
  * @tparam Predicate Unary predicate
  *
  * @param[in] map_ref Reference of the map into which inserts will be performed
  * @param[in] key_begin The beginning of the range of keys to insert
  * @param[in] value_begin The beginning of the range of values associated with each key to insert
  * @param[in] num_keys The total number of keys and values
  * @param[in] pred Unary predicate applied to each key. Only keys that pass the predicated will be
  * inserted.
  * @param[out] num_inserted The total number of keys successfully inserted
  */
 template <typename Map, typename KeyIter, typename ValueIter>
 __global__ void insert(Map map_ref,
                        KeyIter key_begin,
                        ValueIter value_begin,
                        size_t num_keys,
                        int* num_inserted)
 {
   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
 
   std::size_t counter = 0;
   while (tid < num_keys) {
      // Map::insert returns `true` if it is the first time the given key was
      // inserted and `false` if the key already existed
      if (map_ref.insert(cuco::pair{key_begin[tid], value_begin[tid]})) {
        ++counter;  // Count number of successfully inserted keys
      }
      tid += gridDim.x * blockDim.x;
   }
 
   // Update global count of inserted keys
   atomicAdd(num_inserted, counter);
 }
 
 /**
  * @brief For keys that have a match in the map, increments their corresponding value by one.
  *
  * @tparam Map Type of the map device reference
  * @tparam KeyIter Input iterator whose value_type convertible to Map::key_type
  *
  * @param map_ref Reference of the map into which queries will be performed
  * @param key_begin The beginning of the range of keys to query
  * @param num_keys The total number of keys
  */
 template <typename Map, typename KeyIter>
 __global__ void increment_values(Map map_ref, KeyIter key_begin, size_t num_keys)
 {
   auto tid = threadIdx.x + blockIdx.x * blockDim.x;
   printf("increment_values: start\n");
   while (tid < num_keys) {
     // If the key exists in the map, find returns an iterator to the specified key. Otherwise it
     // returns map.end()
     auto found = map_ref.find(key_begin[tid]);
     if (found != map_ref.end()) {
       // If the key exists, atomically increment the associated value
       auto ref =
         cuda::atomic_ref<typename Map::mapped_type, cuda::thread_scope_device>{found->second};
       ref.fetch_add(1, cuda::memory_order_relaxed);
     }
     tid += gridDim.x * blockDim.x;
   }
   printf("increment_values: end\n");
 }

 template <typename ValueIter>
 __global__ void insert_vector_pointers(ValueIter insert_values_begin, 
                                        int* pointer_underl_values_begin, 
                                        size_t num_keys, int* num_inserted) {
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    pointer_underl_values_begin += tid;

    std::size_t counter = 0;
    while (tid < num_keys) {
        if (insert_values_begin[tid] = pointer_underl_values_begin) {
            counter++;
            printf("%d %d\n\n", tid, *insert_values_begin[tid]);
        }
        pointer_underl_values_begin += gridDim.x * blockDim.x;
        tid += gridDim.x * blockDim.x;
    }
    atomicAdd(num_inserted, counter);
 }

 
 int main(void)
 { 
   // Empty slots are represented by reserved "sentinel" values. These values should be selected such
   // that they never occur in your input data.
   auto const empty_idx4_sentinel = idx4{uint16NaN, uint16NaN, uint16NaN, uint16NaN};
   Cell* empty_pcell_sentinel = nullptr;
   int* empty_pvalue_sentinel = nullptr;
   Cell empty_cell_sentinel = Cell{doubleNaN, doubleNaN, doubleNaN, doubleNaN, int32_tNaN};

   // Number of key/value pairs to be inserted
   size_t constexpr num_keys = 100;
   int constexpr empty_value_sentinel = -1;
   
   auto constexpr block_size = 256;
   auto const grid_size      = (num_keys + block_size - 1) / block_size;
 
 
   // Create a sequence of keys and values {{0,0}, {1,1}, ... {i,i}}
   // device can't allocate on device
   thrust::device_vector<idx4> insert_keys(num_keys);
   thrust::device_vector<int>  underlying_values(num_keys); // Cell
   thrust::device_vector<int*> insert_values(num_keys); // int* -> vector[1] of value
   //int ptr = thrust::raw_pointer_cast(value.data());
   for (uint16 i = 0; i < num_keys; i++) {
        insert_keys[i] = idx4{i, i+1, i+2, i+3};
        underlying_values[i] = i*2;
        //insert_values[i] = ptr;
        //ptr++;
   }

   // Allocate storage for count of number of inserted keys
   thrust::device_vector<int> num_inserted(1);

   insert_vector_pointers<<<grid_size, block_size>>>(insert_values.begin(), 
                                                     thrust::raw_pointer_cast(underlying_values.data()),
                                                     num_keys,
                                                     num_inserted.data().get());
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

   std::cout << "Number of underlying values inserted: " << num_inserted[0] << std::endl;

   // Compute capacity based on a 100% load factor
   auto constexpr load_factor = 1.0;
   size_t const capacity = NCELL_MAX;
 
   // Constructs a map with "capacity" slots using -1 and -1 as the empty key/value sentinels.
   auto map = cuco::static_map{capacity,
                              cuco::empty_key{empty_idx4_sentinel},
                              cuco::empty_value{empty_pvalue_sentinel}, // empty_pcell_sentinel
                              thrust::equal_to<idx4>{},
                              cuco::linear_probing<1, cuco::default_hash_function<idx4>>{}};
                              //idx4_equals{},
                              //cuco::linear_probing<1, ramses_hash>{}};
 
   // Get a non-owning, mutable reference of the map that allows inserts to pass by value into the
   // kernel
   auto insert_ref = map.ref(cuco::insert);
 
   // Predicate will only insert even keys
   // auto is_even = [] __device__(auto key) { return (key % 2) == 0; };
 
   // reset num_inserted
   num_inserted[0] = 0;
 
   insert<<<grid_size, block_size>>>(insert_ref,
                                    insert_keys.begin(),
                                    insert_values.begin(),
                                    num_keys,
                                    num_inserted.data().get());

   cudaDeviceSynchronize();
   CHECK_LAST_CUDA_ERROR();
 
   std::cout << "Number of keys inserted: " << num_inserted[0] << std::endl;
 
   // Get a non-owning reference of the map that allows find operations to pass by value into the
   // kernel
   auto find_ref = map.ref(cuco::find);
 
   /*increment_values<<<grid_size, block_size>>>(find_ref, insert_keys.begin(), num_keys);

   cudaDeviceSynchronize();
   CHECK_LAST_CUDA_ERROR();

   cout << "Passed increment_values\n";*/
 
   // Retrieve contents of all the non-empty slots in the map
   thrust::device_vector<idx4> contained_keys(num_inserted[0]);
   thrust::device_vector<int*> contained_values(num_inserted[0]); // Cell
   map.retrieve_all(contained_keys.begin(), contained_values.begin());
 
   auto tuple_iter =
     thrust::make_zip_iterator(thrust::make_tuple(contained_keys.begin(), contained_values.begin()));
   // Iterate over all slot contents and verify that `slot.key + 1 == slot.value` is always true.

   cout << "Starting thrust::all_of\n";
   auto result = thrust::all_of(
     thrust::device, tuple_iter, tuple_iter + num_inserted[0], [] __device__(auto const& tuple) {
       return thrust::get<0>(tuple).idx3[0]*2 == *thrust::get<1>(tuple); // thrust::get<1>(tuple).rho
     });
 
   if (result) { cout << "Success! Target values are properly retrieved.\n"; } //incremented
 
   return 0;
 }