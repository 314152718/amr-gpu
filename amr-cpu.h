
const int L_base = 3;
const int L_max = 6;
const int N_dim = 3;
const int N_cell_max = 10000;
const double rho_crit = 0.1;

struct idx4 {
    unsigned int i, j, k, L;
};
struct Cell {
    unsigned int i, j, k, L;
    double rho;
    int flag_leaf;
    int flag_faces[6];
};

unsigned int transposeToHilbert(const unsigned int X[N_dim], const int L);
void hilbertToTranspose(unsigned int X[N_dim], const unsigned int h, const int L);
void getHindex(idx4 idx_cell, unsigned int& hindex);
void getHindexInv(unsigned int hindex, int L, idx4& cell_idx);
void makeBaseGrid();
double rhoFunc(double x, double y, double z, double sigma = 1.0);
