#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <cufft.h>
#include <cuda.h>
#include <math.h>
#include <cuComplex.h>
#include <time.h>
// #include <omp.h>
/* #define M_PI acos(-1.0) */

void myDST_1DMULT(int Nx, int Ny, double *b, double *bhat);

__global__ void solveSystem(int Nx, int Ny, double *rhat,double *y_a, double *xhat, double *d_DL, double * d_DU){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < Nx){

        y_a[i*Ny] = rhat[i];

        int mx = i*Ny ;
        for(int j = 1; j < Ny; j++){
            int my = j*Nx;
            y_a[j + mx] = rhat[i + my] - d_DL[j + mx]*y_a[j - 1 + mx];

        }

        xhat[i+(Ny-1)*Nx] = y_a[Ny - 1 + mx]/d_DU[Ny - 1 + mx] ;

        for(int j = Ny-2; j >= 0; j--){

            xhat[i+j*Nx] =  ( y_a[j + mx] - xhat[i+(j+1)*Nx] )/d_DU[j + mx] ;


        }
    }
}

void solver_2D_dir_Y_DD_overlap(int Nx, int Ny, double *SOL,
                                double *DL,double *DU, double *rhs, int number_of_threads)

{
/*
  This program is a 2D direct solver of Poisson equation
  The solution is based on 1D Fast Fourier Transform and
  1-D direct solver in the vertical direction.
  To have a close to the optimal solution we need to have
  m = 2^(n) - 1 , n=1,2,... dimensional vector.
 ========================================================================
  input:  Nx,Ny     - the number of grid points in x- and y-directions.
  		  rhs 		- an Nx*Ny right hand side vector
          DL   		- the subdiagonal in the L matrix from the LU
                      factorization of the transformed tridiagonal matrix.
          UL  		- the main diagonal in the U matrix from the LU
                	  factorization of the transformed tridiagonal matrix.
  output: SOL 		- is the solution of Ax=rhs

 ========================================================================

  Dr. Yury Gryazin, 03/01/2017, ISU, Pocatello, ID
*/

    int Nxy ;
    double *rhat, *xhat, *d_rhs, *d_DL, *d_DU, *d_SOL;

/*
   Allocate the memory for the transformed rhs (rhat), for the auxiliary
   solutin vector y_a and for the solution vector of the transformed system.
*/
    Nxy = Nx*Ny;

    //printf("before cudaMalloc\n");
	cudaMalloc((void**)&rhat,Nxy * sizeof(double));
    cudaMalloc((void**)&xhat,Nxy * sizeof(double));
    cudaMalloc((void**)&d_rhs,Nxy * sizeof(double));
    cudaMalloc((void**)&d_DL,Nxy * sizeof(double));
    cudaMalloc((void**)&d_DU,Nxy * sizeof(double));
    cudaMalloc((void**)&d_SOL,Nxy * sizeof(double));

/*
   FFT transformation of the rhs layer by layer (y = const).
   and store the transformed vectors in the rhat in the transpose order
   (vertically) in contrast to rhs.
*/
    //printf("before Memcopy\n");
    cudaMemcpy(d_rhs, rhs, sizeof(double) * Nxy,cudaMemcpyHostToDevice);
    cudaMemcpy(d_DU, DU, sizeof(double) * Nxy,cudaMemcpyHostToDevice);
    cudaMemcpy(d_DL, DL, sizeof(double) * Nxy,cudaMemcpyHostToDevice);


    myDST_1DMULT(Nx, Ny, d_rhs, rhat);
    //printf("successful DST call\n");
/*
   Solution of the transformed system.
*/


{

    double *y_a;
    cudaMalloc((void**)&y_a,Nxy * sizeof(double));

// #pragma omp for private(mx,my,j)
    dim3 block (512);
    dim3 grid  ((Nx + block.x - 1) / block.x);
    solveSystem<<<grid,block>>>( Nx,  Ny, rhat,y_a, xhat, d_DL, d_DU );

/*
   Inverse FFT transformation of the solution of the transformed system.
*/

/*
   FFT transformation of the xhat layer by layer (y = const).
   and store the transformed vectors in the original order as in rhs.
*/




    myDST_1DMULT(Nx, Ny, xhat, d_SOL);

    cudaMemcpy(SOL,d_SOL,sizeof(double) *Nxy,cudaMemcpyDeviceToHost);

    cudaFree(y_a);
    y_a   = NULL;


}
    cudaFree(rhat);
     rhat   = NULL;
    cudaFree(xhat);
     xhat   = NULL;
     cudaFree(d_rhs);
     d_rhs = NULL;
     cudaFree(d_DU);
     d_DU = NULL;
     cudaFree(d_DL);
     d_DL = NULL;
     cudaFree(d_SOL);
     d_SOL = NULL;

    return;
}
