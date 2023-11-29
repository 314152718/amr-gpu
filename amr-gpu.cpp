#include <iostream>
#include <vector>
#include <string>
#include "amr-cpu.cpp"

using namespace std;


int main() {
    unsigned int X[3] = {20, 10, 30};
    hindex(X, 5, N_dim);
    cout << X[0] << endl;
}