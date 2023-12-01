
typedef unsigned int uint;

const uint L_base = 3;
const uint L_max = 6;
const uint N_dim = 3;
const uint N_cell_max = 10000;
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
const int hash_constants[4] = {-1640531527, 97, 1003313, 5};

struct idx4 {
    uint idx3[N_dim];
    uint L;
    bool operator==(const idx4 &other) const {
        return idx3[0] == other.idx3[0] && idx3[1] == other.idx3[1] && idx3[2] == other.idx3[2] && L == other.L;
    }
};

struct Cell {
    double rho;
    bool flag_leaf;
    double rho_grad[3];
};

void transposeToHilbert(const unsigned int X[N_dim], const int L, uint &hindex);
void hilbertToTranspose(const uint hindex, const int L, uint (&X)[N_dim]);
void getHindex(idx4 idx_cell, uint& hindex);
void getHindexInv(uint hindex, int L, idx4& idx_cell);

double rhoFunc(const double coord[N_dim], const double sigma = 1.0);
bool refCrit(double rho);

void getParentIdx(const idx4 idx_cell, idx4 idx_parent);
void getNeighborIdx(const idx4 idx_cell, const uint dir, const bool pos, idx4 &idx_neighbor);
bool checkIfExists(const idx4 idx_cell);
void checkIfBorder(const idx4 idx_cell, const uint dir, const bool pos, bool &is_border);

void makeBaseGrid(Cell (&grid)[N_cell_max]);
void setGridCell(const idx4 idx_cell, const uint hindex);
void refineGridCell(const idx4 idx_cell);

void getNeighborInfo(const idx4 idx_cell, const uint dir, const bool pos, bool &is_ref, double &rho);
void calcGradCell(idx4 idx_cell);
void calcGrad();
