
const int L_base = 3;
const int L_max = 6;
const int N_dim = 3;
const int N_cell_max = 10000;
const double rho_crit = 0.1;

struct idx4 {
    uint idx3[N_dim];
    uint L;
};
struct Cell {
    double rho;
    bool flag_leaf;
};

unsigned int transposeToHilbert(const unsigned int X[N_dim], const int L);
void hilbertToTranspose(unsigned int X[N_dim], const unsigned int h, const int L);
void getHindex(idx4 idx_cell, unsigned int& hindex);
void getHindexInv(unsigned int hindex, int L, idx4& cell_idx);
void makeBaseGrid();
double rhoFunc(const double coord[N_dim], double sigma = 1.0);
