#include <complex.h>
#define INDEX(i,j,Nx) i+j*Nx

void residual2D(int Nx, int Ny, double hx, double hy,
					double *SOL, double *rhs, double *res )
                
{
/*
 This program is a 2D matrix - vector multiplication
 where the matrix A is the discretization of the Poisson equation.

========================================================================
 input:  Nx,Ny      - the number of grid points in x- and y-directions.
  		 hx,hy	- the grid sizes in x- and y-directions.
		 SOL 		- Nx*Ny vector
		 rhs		- the right hand side
 output: res 		- A*SOL-rhs vector

========================================================================
  Dr. Yury Gryazin, 03/02/2017, ISU, Pocatello, ID
*/


int i,j;
double Ryx;
double LXX, LYY;

	Ryx = hy*hy/hx/hx;
    
    for( j = 0; j < Ny; j++) {
        for( i = 0; i < Nx; i++){
        
            if(i == 0)
                LXX = SOL[INDEX((i+1),j,Nx)] ;
            else {
                if(i == Nx-1)
                    LXX = SOL[INDEX((i-1),j,Nx)] ;
                else
                    LXX = SOL[INDEX((i-1),j,Nx)] + SOL[INDEX((i+1),j,Nx)]  ;
            }
            if(j == 0)
                LYY = SOL[INDEX(i,(j+1),Nx)] ;
            else {
                if(j == Ny-1)
                    LYY = SOL[INDEX(i,(j-1),Nx)] ;
                else
                    LYY = SOL[INDEX(i,(j-1),Nx)] + SOL[INDEX(i,(j+1),Nx)]  ;
            }
                 
            res[INDEX(i,j,Nx)] = Ryx*LXX + LYY + (-2-2*Ryx)*SOL[INDEX(i,j,Nx)]
                                 - rhs[INDEX(i,j,Nx)];
    				
	    		
			}	
    }
				
	

return;
}
