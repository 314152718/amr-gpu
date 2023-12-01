
typedef unsigned int uint;

const int L_base = 3;
const int L_max = 6;
const int N_dim = 3;
const int N_cell_max = 10000;
const double rho_crit = 0.1;
const double rho_boundary = 0.; // boundary condition

/*
Finite difference kernel
df/dx = [ col0 * f(x_0) + col1 * f(x_1) + col2 * f(x_2) ] / col3
row0: x_0 = -1.5, x_1 = 0, x_2 = 1.5
row1: x_0 = -1, x_1 = 0, x_2 = 1.5
row2: x_0 = -1.5, x_1 = 0, x_2 = 1
row3: x_0 = -1, x_1 = 0, x_2 = 1
*/
const int fd_kernel[4][4] = {
    {-1, 0, 1, 3},
    {-9, 5, 4, 15},
    {-4, -5, 9, 15},
    {-1, 0, 1, 2}
};

struct idx4 {
    uint idx3[N_dim];
    uint L;
};

struct Cell {
    double rho;
    bool flag_leaf;
    double rho_grad[3];
};

unsigned int transposeToHilbert(const unsigned int X[N_dim], const int L);
void hilbertToTranspose(unsigned int X[N_dim], const unsigned int h, const int L);
void getHindex(idx4 idx_cell, unsigned int& hindex);
void getHindexInv(unsigned int hindex, int L, idx4& cell_idx);
void makeBaseGrid();
double rhoFunc(const double coord[N_dim], const double sigma = 1.0);
