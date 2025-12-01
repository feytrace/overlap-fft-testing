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


// Cosine kernel computation: K(x,x') = σ² × cos(k0_x(x-x')) × cos(k0_y(y-y'))
__global__ void compute_cosine_kernel_matrix(double *K, double *X, double *Y, 
                                              int N, double sigma2, 
                                              double k0_x, double k0_y) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N || j >= N) return;
    
    // Compute K(i,j)
    double dx = X[i] - X[j];
    double dy = Y[i] - Y[j];
    
    double cos_x = cos(k0_x * dx);
    double cos_y = cos(k0_y * dy);
    
    K[i * N + j] = sigma2 * cos_x * cos_y;
}

// Add noise to diagonal (K = K + σ²_noise * I)
__global__ void add_noise_diagonal(double *K, int N, double noise_var) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    K[idx * N + idx] += noise_var;
}

// Cholesky decomposition kernel (sequential, but can be used per stream chunk)
__global__ void cholesky_decomposition_chunk(double *L, int N, int start_row, int end_row) {
    // This is a simplified version - full Cholesky needs careful synchronization
    // For overlapping, we'll break this into row-wise chunks
    
    int tid = threadIdx.x;
    
    for (int i = start_row; i < end_row && i < N; i++) {
        // Compute L[i,i]
        if (tid == 0) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                double val = L[i * N + k];
                sum += val * val;
            }
            L[i * N + i] = sqrt(L[i * N + i] - sum);
        }
        __syncthreads();
        
        // Compute L[j,i] for j > i
        for (int j = i + 1 + tid; j < N; j += blockDim.x) {
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += L[j * N + k] * L[i * N + k];
            }
            L[j * N + i] = (L[j * N + i] - sum) / L[i * N + i];
        }
        __syncthreads();
    }
}

// Forward substitution: solve Ly = b for y
__global__ void forward_substitution(double *L, double *b, double *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    double sum = 0.0;
    for (int j = 0; j < idx; j++) {
        sum += L[idx * N + j] * y[j];
    }
    y[idx] = (b[idx] - sum) / L[idx * N + idx];
}

// Backward substitution: solve L^T x = y for x
__global__ void backward_substitution(double *L, double *y, double *x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N) return;
    
    int i = N - 1 - idx;
    
    double sum = 0.0;
    for (int j = i + 1; j < N; j++) {
        sum += L[j * N + i] * x[j];
    }
    x[i] = (y[i] - sum) / L[i * N + i];
}

// Compute predictive mean: μ* = K*^T × K^{-1} × y
__global__ void predict_mean_kernel(double *mu_pred, double *K_star, 
                                     double *alpha, int N_train, int N_test) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N_test) return;
    
    double sum = 0.0;
    for (int j = 0; j < N_train; j++) {
        sum += K_star[i * N_train + j] * alpha[j];
    }
    mu_pred[i] = sum;
}

// Compute K_star: covariance between test and training points
__global__ void compute_K_star(double *K_star, 
                                double *X_test, double *Y_test,
                                double *X_train, double *Y_train,
                                int N_test, int N_train,
                                double sigma2, double k0_x, double k0_y) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // test point
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // train point
    
    if (i >= N_test || j >= N_train) return;
    
    double dx = X_test[i] - X_train[j];
    double dy = Y_test[i] - Y_train[j];
    
    double cos_x = cos(k0_x * dx);
    double cos_y = cos(k0_y * dy);
    
    K_star[i * N_train + j] = sigma2 * cos_x * cos_y;
}

// Compute predictive variance: σ²* = K** - K*^T K^{-1} K*
__global__ void predict_variance_kernel(double *var_pred, double *K_star_star,
                                         double *K_star, double *v,
                                         int N_test, int N_train) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= N_test) return;
    
    double sum = 0.0;
    for (int j = 0; j < N_train; j++) {
        sum += K_star[i * N_train + j] * v[i * N_train + j];
    }
    
    var_pred[i] = K_star_star[i] - sum;
}


class GPCosineKernel {
public:
    int N_train;
    double sigma2;
    double k0_x, k0_y;
    double noise_var;
    
    double *d_X_train, *d_Y_train;
    double *d_f_train;  // training outputs
    double *d_K;        // covariance matrix
    double *d_L;        // Cholesky factor
    double *d_alpha;    // K^{-1} * f
    
    cudaStream_t *streams;
    int nStreams;
    
    GPCosineKernel(int n, double sig2, double kx, double ky, double noise, int n_streams = 4) {
        N_train = n;
        sigma2 = sig2;
        k0_x = kx;
        k0_y = ky;
        noise_var = noise;
        nStreams = n_streams;
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_X_train, N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Y_train, N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_f_train, N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_K, N_train * N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_L, N_train * N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_alpha, N_train * sizeof(double)));
        
        // Create streams
        streams = new cudaStream_t[nStreams];
        for (int i = 0; i < nStreams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~GPCosineKernel() {
        CUDA_CHECK(cudaFree(d_X_train));
        CUDA_CHECK(cudaFree(d_Y_train));
        CUDA_CHECK(cudaFree(d_f_train));
        CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_L));
        CUDA_CHECK(cudaFree(d_alpha));
        
        for (int i = 0; i < nStreams; i++) {
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
        }
        delete[] streams;
    }
    
    void fit(double *X_train, double *Y_train, double *f_train) {
        // Transfer training data to device with overlapping
        int chunk_size = (N_train + nStreams - 1) / nStreams;
        
        for (int s = 0; s < nStreams; s++) {
            int start = s * chunk_size;
            int end = min(start + chunk_size, N_train);
            int size = end - start;
            
            if (start >= N_train) break;
            
            CUDA_CHECK(cudaMemcpyAsync(&d_X_train[start], &X_train[start],
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(&d_Y_train[start], &Y_train[start],
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(&d_f_train[start], &f_train[start],
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
        }
        
        // Compute kernel matrix K
        dim3 block(16, 16);
        dim3 grid((N_train + block.x - 1) / block.x,
                  (N_train + block.y - 1) / block.y);
        
        compute_cosine_kernel_matrix<<<grid, block>>>(d_K, d_X_train, d_Y_train,
                                                       N_train, sigma2, k0_x, k0_y);
        CUDA_CHECK(cudaGetLastError());
        
        // Add noise to diagonal
        int threads = 256;
        int blocks = (N_train + threads - 1) / threads;
        add_noise_diagonal<<<blocks, threads>>>(d_K, N_train, noise_var);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy K to L for Cholesky
        CUDA_CHECK(cudaMemcpy(d_L, d_K, N_train * N_train * sizeof(double),
                              cudaMemcpyDeviceToDevice));
        
        // Cholesky decomposition (simplified - for production use cuSOLVER)
        // Here we do it on CPU for stability
        double *h_L = new double[N_train * N_train];
        CUDA_CHECK(cudaMemcpy(h_L, d_L, N_train * N_train * sizeof(double),
                              cudaMemcpyDeviceToHost));
        
        // CPU Cholesky
        for (int i = 0; i < N_train; i++) {
            for (int j = 0; j < i; j++) {
                double sum = 0.0;
                for (int k = 0; k < j; k++) {
                    sum += h_L[i * N_train + k] * h_L[j * N_train + k];
                }
                h_L[i * N_train + j] = (h_L[i * N_train + j] - sum) / h_L[j * N_train + j];
            }
            
            double sum = 0.0;
            for (int k = 0; k < i; k++) {
                sum += h_L[i * N_train + k] * h_L[i * N_train + k];
            }
            h_L[i * N_train + i] = sqrt(h_L[i * N_train + i] - sum);
        }
        
        CUDA_CHECK(cudaMemcpy(d_L, h_L, N_train * N_train * sizeof(double),
                              cudaMemcpyHostToDevice));
        delete[] h_L;
        
        // Solve for alpha = K^{-1} * f
        // First solve Ly = f
        double *d_y;
        CUDA_CHECK(cudaMalloc(&d_y, N_train * sizeof(double)));
        
        // CPU forward/backward substitution for stability
        double *h_L_final = new double[N_train * N_train];
        double *h_f = new double[N_train];
        double *h_y = new double[N_train];
        double *h_alpha = new double[N_train];
        
        CUDA_CHECK(cudaMemcpy(h_L_final, d_L, N_train * N_train * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_f, d_f_train, N_train * sizeof(double),
                              cudaMemcpyDeviceToHost));
        
        // Forward substitution: Ly = f
        for (int i = 0; i < N_train; i++) {
            double sum = 0.0;
            for (int j = 0; j < i; j++) {
                sum += h_L_final[i * N_train + j] * h_y[j];
            }
            h_y[i] = (h_f[i] - sum) / h_L_final[i * N_train + i];
        }
        
        // Backward substitution: L^T alpha = y
        for (int i = N_train - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < N_train; j++) {
                sum += h_L_final[j * N_train + i] * h_alpha[j];
            }
            h_alpha[i] = (h_y[i] - sum) / h_L_final[i * N_train + i];
        }
        
        CUDA_CHECK(cudaMemcpy(d_alpha, h_alpha, N_train * sizeof(double),
                              cudaMemcpyHostToDevice));
        
        delete[] h_L_final;
        delete[] h_f;
        delete[] h_y;
        delete[] h_alpha;
        CUDA_CHECK(cudaFree(d_y));
    }
    
    void predict(double *X_test, double *Y_test, int N_test,
                 double *mu_pred, double *var_pred) {
        // Allocate device memory for test data
        double *d_X_test, *d_Y_test;
        double *d_K_star;
        double *d_mu_pred, *d_var_pred;
        
        CUDA_CHECK(cudaMalloc(&d_X_test, N_test * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_Y_test, N_test * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_K_star, N_test * N_train * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_mu_pred, N_test * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_var_pred, N_test * sizeof(double)));
        
        // Transfer test data with overlapping
        int chunk_size = (N_test + nStreams - 1) / nStreams;
        
        for (int s = 0; s < nStreams; s++) {
            int start = s * chunk_size;
            int end = min(start + chunk_size, N_test);
            int size = end - start;
            
            if (start >= N_test) break;
            
            CUDA_CHECK(cudaMemcpyAsync(&d_X_test[start], &X_test[start],
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
            CUDA_CHECK(cudaMemcpyAsync(&d_Y_test[start], &Y_test[start],
                                       size * sizeof(double),
                                       cudaMemcpyHostToDevice, streams[s]));
        }
        
        // Compute K_star
        dim3 block(16, 16);
        dim3 grid((N_train + block.x - 1) / block.x,
                  (N_test + block.y - 1) / block.y);
        
        compute_K_star<<<grid, block>>>(d_K_star, d_X_test, d_Y_test,
                                         d_X_train, d_Y_train,
                                         N_test, N_train,
                                         sigma2, k0_x, k0_y);
        CUDA_CHECK(cudaGetLastError());
        
        // Compute predictive mean
        int threads = 256;
        int blocks = (N_test + threads - 1) / threads;
        predict_mean_kernel<<<blocks, threads>>>(d_mu_pred, d_K_star, d_alpha,
                                                  N_train, N_test);
        CUDA_CHECK(cudaGetLastError());
        
        // Transfer results back with overlapping
        for (int s = 0; s < nStreams; s++) {
            int start = s * chunk_size;
            int end = min(start + chunk_size, N_test);
            int size = end - start;
            
            if (start >= N_test) break;
            
            CUDA_CHECK(cudaMemcpyAsync(&mu_pred[start], &d_mu_pred[start],
                                       size * sizeof(double),
                                       cudaMemcpyDeviceToHost, streams[s]));
        }
        
        // Synchronize all streams
        for (int s = 0; s < nStreams; s++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_X_test));
        CUDA_CHECK(cudaFree(d_Y_test));
        CUDA_CHECK(cudaFree(d_K_star));
        CUDA_CHECK(cudaFree(d_mu_pred));
        CUDA_CHECK(cudaFree(d_var_pred));
    }
};


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}


int main(int argc, char **argv) {
    // GP hyperparameters
    double sigma2 = 1.0;      // Signal variance
    double k0_x = 2.0 * M_PI; // Frequency in x
    double k0_y = 2.0 * M_PI; // Frequency in y
    double noise_var = 0.01;  // Noise variance
    
    int N_train = 100;
    int N_test = 50;
    int nStreams = 4;
    
    if (argc > 1) nStreams = atoi(argv[1]);
    if (argc > 2) N_train = atoi(argv[2]);
    if (argc > 3) N_test = atoi(argv[3]);
    
    printf("\n========================================\n");
    printf("Gaussian Process with Cosine Kernel\n");
    printf("========================================\n");
    printf("Training points: %d\n", N_train);
    printf("Test points: %d\n", N_test);
    printf("Streams: %d\n", nStreams);
    printf("Kernel: K(x,x') = %.2f × cos(%.2f(x-x')) × cos(%.2f(y-y'))\n",
           sigma2, k0_x, k0_y);
    printf("Noise variance: %.4f\n", noise_var);
    
    // Generate training data
    double *X_train = new double[N_train];
    double *Y_train = new double[N_train];
    double *f_train = new double[N_train];
    
    srand(42);
    for (int i = 0; i < N_train; i++) {
        X_train[i] = (double)rand() / RAND_MAX;
        Y_train[i] = (double)rand() / RAND_MAX;
        // True function: f(x,y) = sin(x) * sin(2y)
        f_train[i] = sin(X_train[i]) * sin(2.0 * Y_train[i]);
        // Add noise
        f_train[i] += (double)rand() / RAND_MAX * 0.1 - 0.05;
    }
    
    // Generate test data
    double *X_test = new double[N_test];
    double *Y_test = new double[N_test];
    double *mu_pred = new double[N_test];
    double *var_pred = new double[N_test];
    
    for (int i = 0; i < N_test; i++) {
        X_test[i] = (double)i / (N_test - 1);
        Y_test[i] = 0.5;
    }
    
    // Create GP model
    printf("\nInitializing GP model...\n");
    GPCosineKernel gp(N_train, sigma2, k0_x, k0_y, noise_var, nStreams);
    
    // Fit model
    printf("Fitting GP model...\n");
    double t0 = cpuSecond();
    gp.fit(X_train, Y_train, f_train);
    double t1 = cpuSecond();
    printf("Fit time: %.6f sec\n", t1 - t0);
    
    // Predict
    printf("\nMaking predictions...\n");
    t0 = cpuSecond();
    gp.predict(X_test, Y_test, N_test, mu_pred, var_pred);
    t1 = cpuSecond();
    printf("Prediction time: %.6f sec\n", t1 - t0);
    
    // Print some predictions
    printf("\nSample predictions (first 10):\n");
    printf("  x      y      pred    true\n");
    printf("-----------------------------------\n");
    for (int i = 0; i < min(10, N_test); i++) {
        double true_val = sin(X_test[i]) * sin(2.0 * Y_test[i]);
        printf("%.3f  %.3f  %7.4f  %7.4f\n",
               X_test[i], Y_test[i], mu_pred[i], true_val);
    }
    
    printf("\n========================================\n\n");
    
    // Cleanup
    delete[] X_train;
    delete[] Y_train;
    delete[] f_train;
    delete[] X_test;
    delete[] Y_test;
    delete[] mu_pred;
    delete[] var_pred;
    
    return 0;
}
