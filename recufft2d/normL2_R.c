#include <stdio.h>
#include <complex.h>
#include <math.h>

double normL2_R(int N, double *x)
                
{
/*
 This program is L2 norm of the real vector of size N.

========================================================================
 input:  N			- the length of the vector.
		 x	 		- a real vector with N components
 output: normL2 	- L2 norm of the vector

========================================================================
  Dr. Yury Gryazin, 03/01/2017, ISU, Pocatello, ID
*/


	int i;
	double norm;

    
    norm=0;
    
	for( i = 0; i < N; i++) {
		norm += x[i]*x[i] ;
	}
	
	
	norm = sqrt(norm);

return norm;
}
