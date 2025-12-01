#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <cuda.h>
/* #define M_PI acos(-1.0) */

void LU_2nd_ord(int Nx, int Ny, double hx, double hy,
                double *DL, double *DU )
{

/*
  Calculate the LU factorization of the transformed system.

========================================================================
 input:  Nx,Ny    - the number of grid points in x- and y-directions.
         hx,hy    - the grid steps in x- and y-directions.

 output: DL       - the subdiagonal in the L matrix from the LU
                    factorization of the transformed tridiagonal matrix.
         UL       - the main diagonal in the U matrix from the LU
                    factorization of the transformed tridiagonal matrix.
========================================================================

  Dr. Yury Gryazin, 03/01/2017, ISU, Pocatello, ID
*/


int    i,j,mxy ;
double Ryx;
double *eigenValue_x ;


/*
   Introduce the eigenvalues of the matrix on every layer
*/

	cudaMallocHost((void**) &eigenValue_x, Nx * sizeof(double));
    Ryx = hy*hy/hx/hx;

	for( i = 0; i < Nx; i++)
	{

    	eigenValue_x[i] = -2.0 - 4.0*Ryx*sin((i+1)*M_PI/2.0/(Nx+1))*sin((i+1)*M_PI/2.0/(Nx+1));
	}



/*
   Introduce the set of tridiagonal matrices
*/

    for( i = 0; i < Nx; i++){

        mxy = i*Ny;
        DU[mxy] = eigenValue_x[i] ;
        DL[mxy] = 0;
    }


	for( i = 0; i < Nx; i++) {
        mxy = i*Ny;
        for( j = 1; j < Ny; j++){
            DL[j + mxy] = 1.0/DU[j + mxy - 1] ;
            DU[j + mxy] = eigenValue_x[i] - DL[j + mxy];
        }
    }



cudaFreeHost(eigenValue_x);
eigenValue_x = NULL;

return;
}
