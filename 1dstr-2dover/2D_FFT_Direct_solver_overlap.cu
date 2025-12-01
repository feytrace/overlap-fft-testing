#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <cuda.h>
#include <sys/time.h>

void rhs_bc_anltsol(int Nx, int Ny, double hx, double hy, double *rhs, double *UE);
void LU_2nd_ord(int Nx, int Ny, double hx, double hy, double *DL, double *DU);
void solver_2D_dir_Y_DD_overlap(int Nx, int Ny, double *SOL,
                                double *DL,double *DU, double *rhs, int threads);
void residual2D(int Nx, int Ny, double hx, double hy, double *SOL, double *rhs, double *res);
double normL2_R(int N, double *x);
double cpuSecond(void);

int main(int argc, char **argv){
    int Nx = 6000;
    int Ny = 6050;
    int K = Nx*Ny;

    printf("\n Matrix size = %d x %d\n", Nx, Ny);

    double hx = 1.0/(Nx+1);
    double hy = 1.0/(Ny+1);

    double *rhs, *UE, *DL, *DU, *SOL, *res;
    cudaMallocHost(&rhs, K*sizeof(double));
    cudaMallocHost(&UE,  K*sizeof(double));
    cudaMallocHost(&DL,  K*sizeof(double));
    cudaMallocHost(&DU,  K*sizeof(double));
    cudaMallocHost(&SOL, K*sizeof(double));
    cudaMallocHost(&res, K*sizeof(double));

    rhs_bc_anltsol(Nx, Ny, hx, hy, rhs, UE);
    LU_2nd_ord(Nx, Ny, hx, hy, DL, DU);

    double t0 = cpuSecond();
    solver_2D_dir_Y_DD_overlap(Nx, Ny, SOL, DL, DU, rhs, 1);
    double t1 = cpuSecond();

    printf("Overlapped solver time = %.6f sec\n", t1-t0);

    residual2D(Nx, Ny, hx, hy, SOL, rhs, res);
    printf("Residual = %.6e\n", normL2_R(K, res));

    for(int i=0;i<K;i++) res[i] = SOL[i] - UE[i];
    printf("Error = %.6e\n\n", normL2_R(K, res));

    cudaFreeHost(rhs); cudaFreeHost(UE);
    cudaFreeHost(DL);  cudaFreeHost(DU);
    cudaFreeHost(SOL); cudaFreeHost(res);
}
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
