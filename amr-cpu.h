
const int L_base = 3;
const int L_max = 6;
const int N_dim = 3;
const int N_cell_max = 10000;

struct idx4;
struct Cell;
int transposeToHilbert(const unsigned int* X, const int L);
void hilbertToTranspose(unsigned int* X, const unsigned int h, const int L);
void hindex(unsigned int* X, int L, int N_dim);
void hindexInv(unsigned int* X, int L, int N_dim);
void make_base_grid();
double rhoFunc(double x, double y, double z);

Cell grid[N_cell_max];
