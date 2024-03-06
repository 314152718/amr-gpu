
#define _USE_MATH_DEFINES

#include <cstdint>
#include <iostream>
#include <string>
#include <chrono>

// includes
#include <fstream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <stdexcept>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int32_t NDIM = 3; // number of dimensions
const int N_dim = 3; // number of dimensions

/*struct idx4 {
    idx4() = default;
    int idx3[NDIM], L;

    idx4(int i_init, int j_init, int k_init, int L_init) : idx3{i_init, j_init, k_init}, L{L_init} {}
    idx4(const int ijk_init[NDIM], int L_init) : idx3{ijk_init[0], ijk_init[1], ijk_init[2]}, L{L_init} {}
    bool operator==(const idx4 &other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
    string str() const {
        return "["+to_string(idx3[0])+", "+to_string(idx3[1])+", "+to_string(idx3[2])+"](L="+to_string(L)+")";
    }
    idx4 (const idx4 &other)
    {
        this->L = other.L;
        for (short i = 0; i < NDIM; i++) {
            this->idx3[i] = other.idx3[i];
        }
    }
};*/

struct Cell {
    double rho;
    bool flag_leaf;
    double rho_grad[3];
};

struct idx4 {
    idx4() = default;
    int idx3[N_dim];
    int L;
    bool operator==(const idx4 &other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
    idx4 (const idx4 &other)
    {
        this->L = other.L;
        for (short i = 0; i < N_dim; i++) {
            this->idx3[i] = other.idx3[i];
        }
    }
};

const int hash_constants[4] = {-1640531527, 97, 1003313, 5};
namespace std
{
    template<>
    struct hash<idx4> {
        size_t operator()(const idx4& idx_cell) const noexcept {
            int complete_hash = 0;
            for (short i = 0; i < N_dim; i++) {
                complete_hash += idx_cell.idx3[i] * hash_constants[i];
            }
            complete_hash += idx_cell.L * hash_constants[3];
            return complete_hash;
        }
    };
}

void test_unordered_map() {
    unordered_map<idx4, Cell*> map;
    //unordered_map<idx4, Cell*>::iterator hashtable_itr;
    //map[idx4{1, 2, 3, 4}] = &Cell{2.0, 4.0, 6.0, 8.0};
    idx4 key;
    key.L = 4;
    key.idx3[0] = 1;
    key.idx3[1] = 2;
    key.idx3[2] = 3;
    printf("%d \n\n", map.find(key) == map.end());
}

int main() {
    test_unordered_map();
}