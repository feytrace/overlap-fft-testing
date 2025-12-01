/*
 ========================================================================

 This program is a multiple 1D cosine transform of a vector based on FFT
 implemented in the FFTW package.
 ========================================================================
 input:
    Nx             - the size of the vectors;
    Ny             - the number of the vectors;
    b              - the array of vectors (located sequentially);
 output:
    bhat           - the array of transformed vectors;
 ========================================================================

 Dr. Yury Gryazin, 03/03/2017, ISU, Pocatello, ID
 */

// #include <omp.h>
#include <stdio.h>
#include <complex.h>
#include <cufft.h>
#include <cuda.h>
#include <math.h>
#include <cuComplex.h>

__global__ void setZeros(int N, double *v){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N){
        v[i] = (cufftDoubleReal) 0.E0;
    }
}

__global__ void setInput(int Ny, int Nx, int NR, double *d_in, double *b){
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if( j < Ny) {
        int my  = j*Nx;
        int myr = j*NR;
        for( int i = 0; i < Nx; i++){

            d_in[i+1 + myr] =(cufftDoubleReal) b[i + my] ;
        }
    }
}

__global__ void recoverOutput(int Ny, int Nx, int NC, cufftDoubleComplex *d_out, double *bhat){
    double coef = sqrt(2.0/(Nx+1)) ;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if( j < Ny) {
        int my  = j*Nx;
        int myc = j*NC;

        for(int i = 0; i < Nx; i++){

            bhat[i + my] = -coef * cuCimag((cuDoubleComplex)d_out[i+1 + myc]) ;
        }

    }

}


void myDST_1DMULT(int Nx, int Ny, double *b, double *bhat)

{




    int NR = 2*Nx + 2;
    int NC = ( NR / 2 ) + 1;

    const int   rank    = 1;
    const int   howmany = Ny;
    int   nr[]    = {NR};
    const int   istride = 1;    const int ostride = 1;
    const int   idist   = NR;   const int odist   = NC;
    int  inembed[] = {0},            onembed[] = {0};



    //printf("before cudaMalloc\n");
		cufftDoubleReal *d_in;
		cudaMalloc((void**)&d_in,sizeof(cufftDoubleReal) * NR*Ny);
        //printf("d_in allocated\n");
		cufftDoubleComplex *d_out;
		cudaMalloc((void**)&d_out, sizeof(cufftDoubleComplex) * NC*Ny);
        //printf("d_out\n");
        cufftHandle handle;

        cufftPlanMany(&handle,rank, nr, inembed, istride, idist, onembed, ostride, odist, CUFFT_D2Z, howmany);
        //printf("plan\n");
        //printf("before zeros\n");

        dim3 block (512);
        dim3 grid  ((NR*Ny + block.x - 1) / block.x);

        setZeros<<<grid, block>>>(NR*Ny, d_in);
        //printf("zeros set\n");

        grid.x =   ((Ny + block.x - 1) / block.x);
        setInput<<<grid, block>>>(Ny,Nx,NR,d_in,b);
        //printf("input set\n");


        cufftExecD2Z(handle, d_in, d_out);
        //printf("successful exec\n");




        recoverOutput<<<grid, block>>>(Ny,Nx,NC,d_out, bhat);
        //printf("output recovered\n");

    cufftDestroy(handle);
	cudaFree(d_in);
	cudaFree(d_out);




    return ;
}
