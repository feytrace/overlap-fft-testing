#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fftw3.h>
/* #define M_PI acos(-1.0) */

void myDST_1D(int Nx, double *b, double *bhat, fftw_plan p1, double *in1, fftw_complex *out1);

void solver_2D_dir_Y_DD(int Nx, int Ny, double *SOL,
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


int i,j,my,mx,Nxy ;
double *rhat, *xhat;
    
/*
   Allocate the memory for the transformed rhs (rhat), for the auxiliary
   solutin vector y_a and for the solution vector of the transformed system.
*/
    Nxy = Nx*Ny; 

	rhat    = (double*) malloc(Nxy * sizeof(double));
	xhat    = (double*) malloc(Nxy * sizeof(double));
/* 
   FFT transformation of the rhs layer by layer (y = const).
   and store the transformed vectors in the rhat in the transpose order
   (vertically) in contrast to rhs.
*/
    
    int N1 = 2*Nx + 2;
    int NC1 = ( N1 / 2 ) + 1;
    
    
#pragma omp parallel default(none) shared(N1,NC1,Nx,Ny,Nxy,rhat,rhs,xhat,SOL,DL,DU) private (i,j)
    {
        
    double *in1 = (double*) fftw_malloc(sizeof(double) * N1);
    fftw_complex *out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NC1);

    double *b_test   = (double*) malloc(Nx * sizeof(double));
    double *bhat_test= (double*) malloc(Nx * sizeof(double));
    double *y_a      = (double*) malloc(Ny  * sizeof(double));
    fftw_plan p1;
        
#pragma omp critical (make_plan)
        
    {
        p1 = fftw_plan_dft_r2c_1d ( N1, in1, out1, FFTW_ESTIMATE );
    }
        
#pragma omp for private(my,i,j)
	for(j = 0; j < Ny; j++) {
        my = j*Nx ;
        for(i = 0; i < Nx; i++){
            b_test[i] = rhs[i + my] ;
		}
        
		myDST_1D(Nx, b_test, bhat_test, p1, in1, out1);

        for(i = 0; i < Nx; i++){
            rhat[i + my] = bhat_test[i] ;
        }
	}
	
    
/*
   Solution of the transformed system.
*/

    
#pragma omp for private(mx,my,j)
    for(i = 0; i < Nx; i++){
        
        y_a[0] = rhat[i];
    
        mx = i*Ny ;
        for(j = 1; j < Ny; j++){
            my = j*Nx;
            y_a[j] = rhat[i + my] - DL[j + mx]*y_a[j - 1];

        }
    
        xhat[Ny - 1 + mx] = y_a[Ny - 1]/DU[Ny - 1 + mx] ;
  
        for(j = Ny-2; j >= 0; j--){
            
            xhat[j + mx] =  ( y_a[j] - xhat[j + 1 + mx] )/DU[j + mx] ;

            
        }
    }
    

/*
   Inverse FFT transformation of the solution of the transformed system. 
*/

/* 
   FFT transformation of the xhat layer by layer (y = const).
   and store the transformed vectors in the original order as in rhs.
*/
    
  
#pragma omp for private(mx, my)
        
        for(j = 0; j < Ny; j++) {
            my = j*Nx;
            for(i = 0; i < Nx; i++){
                mx = i*Ny ;
                b_test[i] = xhat[j + mx] ;
            }
            
            myDST_1D(Nx, b_test, bhat_test, p1, in1, out1);
            
            for(i = 0; i < Nx; i++){
                SOL[i + my] = bhat_test[i] ;
            }
        }
        
        fftw_destroy_plan(p1);
        fftw_free(in1);
          in1 = NULL;
        fftw_free(out1);
          out1 = NULL;
        free(b_test);
          b_test = NULL;
        free(bhat_test);
          bhat_test = NULL;
        free(y_a);
          y_a   = NULL;

        
    }

free(rhat);
     rhat   = NULL;
free(xhat);
     xhat   = NULL;

return;
}

