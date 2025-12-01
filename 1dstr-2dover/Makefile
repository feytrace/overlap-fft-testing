# Compiler
CC = nvcc

# Flags
CFLAGS = -O3 -g --allow-unsupported-compiler

# Includes (NVIDIA HPC SDK)
INCLUDES = -I/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/REDIST/include

# Libraries
LIBS = -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux/lib -lcufft -lcufftw -lm

# Source files (your overlapped solver will replace the normal one)
SRCS = 2D_FFT_Direct_solver_overlap.cu \
       solver_2D_dir_Y_DD_overlap.cu \
       LU_2nd_ord.cu \
       myDST_1DMULT.cu \
       residual2D.cu \
       rhs_bc_anltsol.cu \
       normL2_R.cu

# Object files
OBJS = $(SRCS:.cu=.o)

# Output binary
MAIN = Direct_solver_2D_overlap

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f *.o *~ $(MAIN)
