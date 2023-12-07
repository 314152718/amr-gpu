# Notes

Storing some useful notes/commands here

### Compiling:
```
nvcc --std=c++17 -arch=sm_70 -I. --expt-relaxed-constexpr --expt-extended-lambda cuco_test.cu -o run
```
- Have to compile for at least architecture `sm_60` (CUCO doesn't work for architectures below that, and the default is `sm_52`)
- need to specify c++17 as well
- adding `-I.` so that all the `#include <cuco/...>` statements are able to find the cuco header files

### Running:
Needs to be run on a node with a GPU. The file `job.slurm` simply runs the executable `./run` generated by `nvcc`.
It is a very basic slurm job right now (could specify output directory in the future)

```
sbatch job.slurm
```

And then inspect the output in the corresponding `slurm-****.out` file that gets created
