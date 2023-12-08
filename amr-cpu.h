// includes
#include <cstdint>
#include <iostream>
#include <string>
#include <chrono>

// namespaces
using namespace std;
using namespace std::chrono;

// constants
const int L_base = 3; // base AMR level
const int L_max = 6; // max AMR level
const int N_dim = 3; // number of dimensions
const int N_cell_max = 2097152 + 10; // (2 ^ (6+1))^3 is how many cells we'd have if all were at level 6. adding 10 to be safe
const double rho_crit = 0.01; // critical density for refinement
const double rho_boundary = 0.; // boundary condition
const double sigma = 0.001; // std of Gaussian density field

/*
Finite difference kernel
df/dx = [ col0 * f(x_0) + col1 * f(x_1) + col2 * f(x_2) ] / col3
row0: x_0 = -1.5, x_1 = 0, x_2 = 1.5
row1: x_0 = -1, x_1 = 0, x_2 = 1.5
row2: x_0 = -1.5, x_1 = 0, x_2 = 1
row3: x_0 = -1, x_1 = 0, x_2 = 1
*/
const double fd_kernel[4][4] = {
    {-1., 0., 1., 3.},
    {-9., 5., 4., 15.},
    {-4., -5., 9., 15.},
    {-1., 0., 1., 2.}
};
const int hash_constants[4] = {-1640531527, 97, 1003313, 5};
const string outfile_name = "grid.csv";
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

ostream& operator<<(ostream &os, const idx4 &idx) {
    os << "[" << idx.idx3[0] << ", " << idx.idx3[1] << ", " << idx.idx3[2] << "](L=" << idx.L << ")";
    return os;
}

struct Cell {
    double rho;
    bool flag_leaf;
    double rho_grad[3];
};

void transposeToHilbert(const unsigned int X[N_dim], const int L, int &hindex);
void hilbertToTranspose(const int hindex, const int L, int (&X)[N_dim]);
void getHindex(idx4 idx_cell, int &hindex);
void getHindexInv(int hindex, int L, idx4 &idx_cell);

double rhoFunc(const double coord[N_dim], const double sigma = 1.0);
bool refCrit(double rho);

void getParentIdx(const idx4 &idx_cell, idx4 &idx_parent);
void getNeighborIdx(const idx4 &idx_cell, const int dir, const bool pos, idx4 &idx_neighbor);
bool checkIfExists(const idx4 &idx_cell);
void checkIfBorder(const idx4 &idx_cell, const int dir, const bool pos, bool &is_border);

void makeBaseGrid(Cell (&grid)[N_cell_max]);
void setGridCell(const idx4 idx_cell, const int hindex, bool flag_leaf);
void setChildrenHelper(idx4 idx_cell, short i);
void refineGridCell(const idx4 idx_cell);

void getNeighborInfo(const idx4 idx_cell, const int dir, const bool pos, bool &is_ref, double &rho_neighbor);
void calcGradCell(const idx4 idx_cell, Cell &cell);
void calcGrad();
void refineGrid1lvl();
void writeGrid();
