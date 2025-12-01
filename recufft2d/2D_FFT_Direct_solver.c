/*
 ========================================================================

 This program is a 2D direct solver of Poisson equation
 u_xx + u_yy = f in [x0,x1]x[y0,y1];
 The solution is based on 1D Fast Fourier Transform and
 1-D direct solver in the vertical direction.
 To have a close to the optimal solution we need to have
 Nx = 2^(n) - 1 , n=1,2,... dimensional vector.
 ========================================================================
 input:
 (defined in the programm)
         Nx,Ny          - the number of grid points in x- and y-directions;
         x0,x1,y0,y1    - the boundaries of the computational domain in x-
                          y-directions;
         rhs            - the right-hand side in the discretized system;
 output: SOL            - is the second order approximation to the solution
                          of the Poisson equation;

 ========================================================================

 Dr. Yury Gryazin, 03/03/2017, ISU, Pocatello, ID
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include <fftw3.h>
#include <sys/time.h>
#include <omp.h>


void rhs_bc_anltsol(int Nx, int Ny, double hx, double hy,
                    double *rhs, double *UE);
void LU_2nd_ord(int Nx, int Ny, double hx, double hy,
                    double *DL, double *DU );
void solver_2D_dir_Y_DD(int Nx, int Ny, double *SOL,
                        double *DL,double *DU, double *rhs, int number_of_threads);
void residual2D(int Nx, int Ny, double hx, double hy,
                double *SOL, double *rhs, double *res );

double normL2_R(int N, double *x);
double normInf_R(int N, double *x);

double cpuSecond(void) ;


int main(int argc, char **argv){

    int number_of_threads = 1;

    if (argc >1) {
        number_of_threads = atoi(argv[1]);
    }

    omp_set_num_threads(number_of_threads);

    int l, Nx, Ny, K;
    double x0,y0,x1,y1,hx,hy,norm;
    double *UE, *rhs;
    double *DL, *DU ;
    double *SOL, *res;
    clock_t time0, time1;
    double start_t, end_t, compute_t;
    double start_t_n, end_t_n, compute_t_n;

/*
     Computational Domain
*/

    x0 = 0;
    y0 = 0;

    x1 = 2*M_PI;
    y1 = 2*M_PI;


    /*
     Nx, Ny - the number of grid points in x- and y- directions,
     K = Nx*Ny - the size of the unknown vector, hx, hy - the grid
     sizes in x- and y- directions.
     */

    Nx=6000;
    Ny=6050;

    if(argc>2){
    	Nx=atoi(argv[2]);
	Ny=atoi(argv[3]);
    }
    printf("\n \tThe size of the matrix is %d by %d \n", Nx, Ny );

    K=Nx*Ny;
    hx=(x1-x0)/(Nx+1);
    hy=(y1-y0)/(Ny+1);


    /*
     Allocate and define analytic solution UE,
     and the corresponding right hand side r.h.s for the test problem.
     */

    UE  = malloc(K * sizeof(double));
    rhs = malloc(K * sizeof(double));

    rhs_bc_anltsol(Nx,Ny,hx,hy,rhs,UE);


    /*
     Find the LU factorization of the transformed system: DL and DU.
     */
    DL    = malloc(K  * sizeof(double));
    DU    = malloc(K  * sizeof(double));


    LU_2nd_ord(Nx, Ny, hx, hy, DL, DU );


    /*
     Direct solution by using Fourier transform in x-direction and
     the direct tridiagonal solver in y-direction.
    */


    time0 = clock();
    start_t = omp_get_wtime();
    start_t_n = cpuSecond();

    SOL  = malloc(K * sizeof(double));

    solver_2D_dir_Y_DD(Nx, Ny, SOL, DL, DU, rhs, number_of_threads);

    end_t = omp_get_wtime();
    end_t_n = cpuSecond();
    time1 = clock();
    compute_t = end_t - start_t;
    compute_t_n = end_t_n - start_t_n;

    printf("\n \tSolver time = %f sec \n", (float)(time1 - time0)/CLOCKS_PER_SEC );
    printf("   \tSolver wall-time = %f sec \n\n", compute_t);
    printf("   \tSolver wall-time (sys/time) = %f sec \n\n", compute_t_n);


    /*
     Calculate the residual A*SOL-rhs
    */
     res  = malloc(K * sizeof(double));

     residual2D(Nx, Ny, hx, hy, SOL, rhs, res );

    /*
     Calculate the l2-norm of the residual
    */
     norm = normInf_R(K, res);

     printf("\t||res||_inf =  %10.7e \n\n",norm);

    /*
     Calculate the error between the approximate and analytic(exact) solution SOL-UE
    */


    for (l=0; l < K; l++){
        res[l] = SOL[l] - UE[l];
    }

     norm = normInf_R(K, res);
     printf("\t||error||_inf =  %10.7e \n\n",norm);


    free(UE);
    UE = NULL;
    free(rhs);
    rhs = NULL;

    free(DL);
    DL = NULL;
    free(DU);
    DU = NULL;
    free(SOL);
    SOL = NULL;
    free(res);
    res = NULL;


    return 0;
} /* end function main */


double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ( (double)tp.tv_sec + (double) tp.tv_usec*1.e-6 );
}
