#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


__global__ void rhs_bc_anltsol_kernel(int Nx, int Ny, double hx, double hy, 
                                       double *rhs, double *UE) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;
    
    if (idx >= total) return;
    
    int i = idx % Nx;
    int j = idx / Nx;
    
    double x = (i + 1) * hx;
    double y = (j + 1) * hy;
    
    const double PI = 3.14159265358979323846;
    
    UE[idx] = sin(PI * x) * sin(PI * y);
    
    rhs[idx] = 2.0 * PI * PI * sin(PI * x) * sin(PI * y);
}

__global__ void LU_setup_kernel(int Nx, int Ny, double hx, double hy,
                                 double *DL, double *DU) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;
    
    if (idx >= total) return;
    
    double ax = 1.0 / (hx * hx);
    double ay = 1.0 / (hy * hy);
    double ac = -2.0 * (ax + ay);
    
    DL[idx] = ac;  
    DU[idx] = ay;  }

// Kernel for tridiagonal forward elimination in Y-direction (per column)
__global__ void tridiagonal_forward_kernel(int Nx, int Ny, double *rhs_mod, 
                                            double *DL_mod, double ay, int col_start, int col_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + col_start;
    
    if (i >= col_end || i >= Nx) return;
    
    // Forward elimination along Y-direction for column i
    for (int j = 1; j < Ny; j++) {
        int idx = j * Nx + i;
        int idx_prev = (j - 1) * Nx + i;
        
        double w = ay / DL_mod[idx_prev];
        DL_mod[idx] = DL_mod[idx] - w * ay;
        rhs_mod[idx] = rhs_mod[idx] - w * rhs_mod[idx_prev];
    }
}

// Kernel for tridiagonal backward substitution in Y-direction (per column)
__global__ void tridiagonal_backward_kernel(int Nx, int Ny, double *SOL, 
                                             double *rhs_mod, double *DL_mod, 
                                             double ay, int col_start, int col_end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + col_start;
    
    if (i >= col_end || i >= Nx) return;
    
    // Backward substitution along Y-direction for column i
    int idx_last = (Ny - 1) * Nx + i;
    SOL[idx_last] = rhs_mod[idx_last] / DL_mod[idx_last];
    
    for (int j = Ny - 2; j >= 0; j--) {
        int idx = j * Nx + i;
        int idx_next = (j + 1) * Nx + i;
        
        SOL[idx] = (rhs_mod[idx] - ay * SOL[idx_next]) / DL_mod[idx];
    }
}

// Kernel for computing residual
__global__ void residual2D_kernel(int Nx, int Ny, double hx, double hy,
                                   double *SOL, double *rhs, double *res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny;
    
    if (idx >= total) return;
    
    int i = idx % Nx;
    int j = idx / Nx;
    
    double ax = 1.0 / (hx * hx);
    double ay = 1.0 / (hy * hy);
    
    double u_center = SOL[idx];
    double laplacian = -2.0 * (ax + ay) * u_center;
    
    // X-direction stencil
    if (i > 0) {
        laplacian += ax * SOL[idx - 1];
    }
    if (i < Nx - 1) {
        laplacian += ax * SOL[idx + 1];
    }
    
    // Y-direction stencil
    if (j > 0) {
        laplacian += ay * SOL[idx - Nx];
    }
    if (j < Ny - 1) {
        laplacian += ay * SOL[idx + Nx];
    }
    
    res[idx] = laplacian - rhs[idx];
}

// Parallel reduction kernel for L2 norm computation
__global__ void reduction_kernel(double *input, double *output, int N) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data and compute square
    sdata[tid] = (idx < N) ? input[idx] * input[idx] : 0.0;
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


void rhs_bc_anltsol(int Nx, int Ny, double hx, double hy, double *rhs, double *UE) {
    int K = Nx * Ny;
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    
    rhs_bc_anltsol_kernel<<<blocks, threads>>>(Nx, Ny, hx, hy, rhs, UE);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void LU_2nd_ord(int Nx, int Ny, double hx, double hy, double *DL, double *DU) {
    int K = Nx * Ny;
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    
    LU_setup_kernel<<<blocks, threads>>>(Nx, Ny, hx, hy, DL, DU);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void solver_2D_dir_Y_DD_overlap(int Nx, int Ny, double *SOL,
                                double *DL, double *DU, double *rhs, int nStreams) {
    int K = Nx * Ny;
    
    double hx = 1.0 / (Nx + 1);
    double hy = 1.0 / (Ny + 1);
    double ay = 1.0 / (hy * hy);
    
    double *d_SOL, *d_DL, *d_DU, *d_rhs;
    double *d_DL_work, *d_rhs_work;  // Working copies for solver
    
    CUDA_CHECK(cudaMalloc(&d_SOL, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_DL, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_DU, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_DL_work, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rhs_work, K * sizeof(double)));
    
    CUDA_CHECK(cudaMemset(d_SOL, 0, K * sizeof(double)));
    
    // Create CUDA streams
    cudaStream_t *streams = new cudaStream_t[nStreams];
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }
    
    // Calculate chunk size for X-direction (columns)
    int chunkCols = (Nx + nStreams - 1) / nStreams;
	
    // HDD Transfer
    for (int s = 0; s < nStreams; s++) {
        int col_start = s * chunkCols;
        int col_end = min(col_start + chunkCols, Nx);
        
        if (col_start >= Nx) break;
        
        int cols_in_chunk = col_end - col_start;
        
        for (int j = 0; j < Ny; j++) {
            int offset = j * Nx + col_start;
            int size = cols_in_chunk;
            
            CUDA_CHECK(cudaMemcpyAsync(&d_DL[offset], &DL[offset], 
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(&d_DU[offset], &DU[offset], 
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(&d_rhs[offset], &rhs[offset], 
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_DL_work, d_DL, K * sizeof(double), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_rhs_work, d_rhs, K * sizeof(double), cudaMemcpyDeviceToDevice));
    
    int threads = 128;
    for (int s = 0; s < nStreams; s++) {
        int col_start = s * chunkCols;
        int col_end = min(col_start + chunkCols, Nx);
        
        if (col_start >= Nx) break;
        
        int cols_in_chunk = col_end - col_start;
        int blocks = (cols_in_chunk + threads - 1) / threads;
        
        tridiagonal_forward_kernel<<<blocks, threads, 0, streams[s]>>>(
            Nx, Ny, d_rhs_work, d_DL_work, ay, col_start, col_end);
    }
    
    // Synchronize before backward pass
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
    
    for (int s = 0; s < nStreams; s++) {
        int col_start = s * chunkCols;
        int col_end = min(col_start + chunkCols, Nx);
        
        if (col_start >= Nx) break;
        
        int cols_in_chunk = col_end - col_start;
        int blocks = (cols_in_chunk + threads - 1) / threads;
        
        tridiagonal_backward_kernel<<<blocks, threads, 0, streams[s]>>>(
            Nx, Ny, d_SOL, d_rhs_work, d_DL_work, ay, col_start, col_end);
    }
    
    for (int s = 0; s < nStreams; s++) {
        int col_start = s * chunkCols;
        int col_end = min(col_start + chunkCols, Nx);
        
        if (col_start >= Nx) break;
        
        int cols_in_chunk = col_end - col_start;
        
        // Transfer solution back for this chunk
        for (int j = 0; j < Ny; j++) {
            int offset = j * Nx + col_start;
            int size = cols_in_chunk;
            
            CUDA_CHECK(cudaMemcpyAsync(&SOL[offset], &d_SOL[offset], 
                                       size * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
    }
    
    // Synchronize all streams
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s]));
    }
    
    // Cleanup streams
    for (int s = 0; s < nStreams; s++) {
        CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
    delete[] streams;
    
    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_SOL));
    CUDA_CHECK(cudaFree(d_DL));
    CUDA_CHECK(cudaFree(d_DU));
    CUDA_CHECK(cudaFree(d_rhs));
    CUDA_CHECK(cudaFree(d_DL_work));
    CUDA_CHECK(cudaFree(d_rhs_work));
}

void residual2D(int Nx, int Ny, double hx, double hy, double *SOL, double *rhs, double *res) {
    int K = Nx * Ny;
    int threads = 256;
    int blocks = (K + threads - 1) / threads;
    
    residual2D_kernel<<<blocks, threads>>>(Nx, Ny, hx, hy, SOL, rhs, res);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

double normL2_R(int N, double *x) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    double *d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(double)));
    
    // First reduction
    reduction_kernel<<<blocks, threads, threads * sizeof(double)>>>(x, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy partial results to host
    double *h_partial = new double[blocks];
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    // Final reduction on CPU
    double sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        sum += h_partial[i];
    }
    
    delete[] h_partial;
    CUDA_CHECK(cudaFree(d_partial));
    
    return sqrt(sum / N);
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char **argv) {
    // Default parameters
    int Nx = 6000;
    int Ny = 6050;
    int nStreams = 4;
    
    // Parse command line arguments
    if (argc > 1) nStreams = atoi(argv[1]);
    if (argc > 2) Nx = atoi(argv[2]);
    if (argc > 3) Ny = atoi(argv[3]);
    
    int K = Nx * Ny;
    
    printf("\n========================================\n");
    printf("CUDA 2D Poisson Solver with Overlapping\n");
    printf("========================================\n");
    printf("Matrix size: %d x %d (%d elements)\n", Nx, Ny, K);
    printf("Number of streams: %d\n", nStreams);
    printf("Memory required: %.2f MB\n", K * sizeof(double) * 6 / (1024.0 * 1024.0));
    
    double hx = 1.0 / (Nx + 1);
    double hy = 1.0 / (Ny + 1);
    
    double *rhs, *UE, *DL, *DU, *SOL, *res;
    CUDA_CHECK(cudaMallocHost(&rhs, K * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&UE, K * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&DL, K * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&DU, K * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&SOL, K * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&res, K * sizeof(double)));
    
    printf("\nInitializing problem...\n");
    
    double t_setup = cpuSecond();
    rhs_bc_anltsol(Nx, Ny, hx, hy, rhs, UE);
    LU_2nd_ord(Nx, Ny, hx, hy, DL, DU);
    printf("Setup time: %.6f sec\n", cpuSecond() - t_setup);
    
    printf("\nSolving system...\n");
    double t0 = cpuSecond();
    solver_2D_dir_Y_DD_overlap(Nx, Ny, SOL, DL, DU, rhs, nStreams);
    double t1 = cpuSecond();
    
    printf("Solver time: %.6f sec\n", t1 - t0);
    printf("Throughput: %.2f GFLOPS\n", (2.0 * K * Ny) / (t1 - t0) / 1e9);
    
    printf("\nComputing residual...\n");
    residual2D(Nx, Ny, hx, hy, SOL, rhs, res);
    double residual = normL2_R(K, res);
    printf("Residual L2 norm: %.6e\n", residual);
    
    for (int i = 0; i < K; i++) {
        res[i] = SOL[i] - UE[i];
    }
    double error = normL2_R(K, res);
    printf("Error L2 norm: %.6e\n", error);
    
    printf("\n========================================\n\n");
    // Save solution to file for visualization
    FILE *fp = fopen("solution.bin", "wb");
    fwrite(SOL, sizeof(double), K, fp);
    fclose(fp);

    // Cleanup
    CUDA_CHECK(cudaFreeHost(rhs));
    CUDA_CHECK(cudaFreeHost(UE));
    CUDA_CHECK(cudaFreeHost(DL));
    CUDA_CHECK(cudaFreeHost(DU));
    CUDA_CHECK(cudaFreeHost(SOL));
    CUDA_CHECK(cudaFreeHost(res));
    
    return 0;
}
