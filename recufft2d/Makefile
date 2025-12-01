
# Define the C compiler to use

CC = cc

# define any compile-time flags
CFLAGS = -Wall -g -fopenmp -O3

# define any directories containing header files other than /usr/include
#
INCLUDES = -I/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-11.4.0/fftw-3.3.10-dtrem5gawc6jc4s6ks3kwekhctegs5vp/include

# define library paths in addition to /usr/lib
#
LFLAGS = -L/global/software/rocky-8.x86_64/gcc/linux-rocky8-x86_64/gcc-11.4.0/fftw-3.3.10-dtrem5gawc6jc4s6ks3kwekhctegs5vp/lib

# define any libraries to link into executable:
#
LIBS = -lfftw3_omp -lfftw3 -lm

# define the C source files
SRCS = 2D_FFT_Direct_solver.c rhs_bc_anltsol.c LU_2nd_ord.c solver_2D_dir_Y_DD.c myDST_1D.c  residual2D.c normL2_R.c normInf_R.c

# define the C object files
#
OBJS = $(SRCS:.c=.o)

# define the executable file
MAIN = Direct_solver_2D


#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all:    $(MAIN)
	@echo  Simple compiler named mycc has been compiled

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
