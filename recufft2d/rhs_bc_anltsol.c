#include <stdlib.h>
#include <math.h>

/* #define M_PI acos(-1.0) */

void rhs_bc_anltsol(int Nx, int Ny, double hx, double hy,
                   double *rhs, double *UE)
{

/* Calculate the r.h.s for the test problem and the corresponding
   analytic solution UE.
========================================================================
 input:  Nx,Ny - the number of grid points in x- and y-directions.
         hx,hy - the grid steps in x- and y-directions,

 output: rhs      - the right hand side of the discretized system,
         UE       - the analytic solution at the grid points,
========================================================================

 Dr. Yury Gryazin, 03/01/2017, ISU, Pocatello, ID
*/


int i,j,my;
double x,y, Ryx;

/*
  The analytic solution and bou
*/
    // Ryx = hy*hy/hx/hx;

    for( j = 0; j < Ny; j++) {
        y=(j+1)*hy;
        my = j*Nx;

        for( i = 0; i < Nx; i++){
            x = (i+1)*hx;

             UE[i + my]  = sin(x)*sin(2*y);
            rhs[i + my]  = hy*hy*(-5)*sin(x)*sin(2*y);
        }
    }

    //
    // for( j = 0; j < Ny; j++){
    //     y = (j+1)*hy;
    //     my = j*Nx;
    //
    //     rhs[my]  = rhs[my] - Ryx*.25*y*(y-1) ;
    //     rhs[my + Nx-1] = rhs[my + Nx-1] - Ryx*.25*y*(y-1) ;
    // }
    //
    // my = (Ny-1)*Nx;
    // for( i = 0; i < Nx; i++){
    //     x = (i+1)*hx;
    //
    //     rhs[i]  = rhs[i] - .25*x*(x-1) ;
    //     rhs[i + my] = rhs[i + my] - .25*x*(x-1) ;
    // }


return;
}
