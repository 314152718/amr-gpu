# Thesis: GPU Acceleration of Adaptive Mesh Refinement using CUDA C++ and Fortran

This folder contains code, experiments, and notes for the thesis project on accelerating Adaptive Mesh Refinement (AMR) through GPU implementations and hash-based data structures.

## Key Thesis Achievements

### 1. GPU-Accelerated AMR via Hashtables
- **Hashtable Implementation:** Implemented a GPU-based hashtable (`cuco::static_map`) for 3D Adaptive Mesh Refinement (AMR) grids, replacing matrix-based storage.
- **Efficiency Proof:** Demonstrated competitive performance against matrix implementations in fully refined grids and superior scaling for non-uniform, sparse astrophysical grid structures.
- **Performance Scaling:** Showed that computation time scales linearly with active cell count, avoiding the memory overhead of storing inactive cells in a full matrix.

### 2. Advanced Performance Optimization
- **Hilbert Space-Filling Curve:** Integrated Hilbert curve ordering for cell indices to improve cache locality and GPU memory coalescence.
  - Achieved ~2x speedup in gradient computations at refinement level $L = 9$.
- **Custom Hash Functions:** Implemented a prime-based hash function inspired by RAMSES, yielding ~1.2x speedup over standard GPU hashing.

### 3. Grace Hopper (GH200) Benchmarking
- **Hardware Acceleration:** Evaluated performance on NVIDIA Grace Hopper GH200 and measured ~2.3x speedup in gradient calculations compared to NVIDIA A100.
- **Memory Interconnect Gains:** Leveraged high-speed CPU-GPU interconnects to achieve ~14.7x speedup in hashtable creation.

### 4. CUDA Development Tooling
- **Static Linter Program:** Developed a Python + `clang.cindex` static linter to detect warp divergence issues in `__device__` functions.
- **Debugging Efficiency:** Provided actionable warnings on inefficient `if-else` blocks and potential performance bottlenecks during development.

## Key files
- `amr-gpu.cu`, `amr-gpu-matrix.cu`, `amr-cpu.cu`: core algorithm implementations (GPU/CPU versions)
- `amr-gpu.h`, `amr-cpu.h`: supporting API/interface headers
- `cuda_test.cu`, `cuco_test.cu`: test harnesses and validation cases
- `custom_type_example.cu`, `host_bulk_example.cu`: utility and example cases
- `graph.py`: plotting/analysis scripts
- `notes.txt`, `notes.sh`: experiment notes and helper commands
- `job_*.slurm`: slurm batch job templates for different clusters
- `README.md`: project README (this file)

## Build and run (standard)
1. Load CUDA module: `module load cudatoolkit/12.2` (or platform equivalent)
2. Build GPU binary:
   - `nvcc --std=c++17 -arch=sm_70 -I. --expt-relaxed-constexpr --expt-extended-lambda cuco_test.cu -o rungpu`
3. Submit a test job using SLURM:
   - `sbatch job.slurm`
4. Inspect output log: `slurm-*.out`

