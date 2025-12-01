/*
 ========================================================================
 
 This program is a 1D cosine transform of a vector based on FFT implemented 
 in the FFTW package.
 ========================================================================
 input:
    Nx             - the size of the vector;
    b              - the vector;
    p1             - real to complex 1D plan for FFTW;
    in1            - input vector for FFTW;
    out1           - output vector for FFTW;
 output: 
    bhat           - transformed vector;
 ========================================================================
 
 Dr. Yury Gryazin, 03/03/2017, ISU, Pocatello, ID
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>
#include <math.h>


void myDST_1D(int Nx, double *b, double *bhat, fftw_plan p1, double *in1, fftw_complex *out1)
    
{    

    int i, N;
    
    double coef;
    
    N = 2*Nx + 2;
    
	for( i = 0; i < N; i++) {
	    in1[i] = 0.0;
    }
    
    for( i = 0; i < Nx; i++){
			
        in1[i+1] = b[i] ;
        }

    fftw_execute(p1);
        
    for( i = 0; i < Nx; i++){
            
        bhat[i] = - cimag(out1[i+1]) ;
    }
    
    coef = sqrt(2.0/(Nx+1)) ;
    
    for( i = 0; i < Nx; i++){
			    
        bhat[i] = coef * bhat[i] ;
    }
				
    return ;
}

