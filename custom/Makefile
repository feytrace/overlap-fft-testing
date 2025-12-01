
# Compiler and flags
NVCC := nvcc
CXX := g++

# sm_50: Maxwell (GTX 9xx, Titan X)
# sm_60: Pascal (GTX 10xx, Titan Xp)
# sm_70: Volta (V100, Titan V)
# sm_75: Turing (RTX 20xx, GTX 16xx)
# sm_80: Ampere (A100, RTX 30xx)
# sm_86: Ampere (RTX 30xx mobile)
# sm_89: Ada Lovelace (RTX 40xx)
# sm_90: Hopper (H100)
CUDA_ARCH := sm_75

NVCC_FLAGS := -O3 -arch=$(CUDA_ARCH) --use_fast_math -Xcompiler -Wall
NVCC_DEBUG_FLAGS := -g -G -arch=$(CUDA_ARCH) -Xcompiler -Wall

# Directories
SRC_DIR := .
OBJ_DIR := obj
BIN_DIR := bin

# Target executable
TARGET := $(BIN_DIR)/cuda_solver
DEBUG_TARGET := $(BIN_DIR)/cuda_solver_debug

# Source files (change this to match your .cu filename)
SOURCES := solver.cu

# Object files
OBJECTS := $(OBJ_DIR)/solver.o
DEBUG_OBJECTS := $(OBJ_DIR)/solver_debug.o

# Default target
.PHONY: all
all: directories $(TARGET)

# Debug target
.PHONY: debug
debug: NVCC_FLAGS = $(NVCC_DEBUG_FLAGS)
debug: directories $(DEBUG_TARGET)

# Create directories
.PHONY: directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Compile release version
$(TARGET): $(OBJECTS)
	@echo "Linking $@..."
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	@echo "Build complete: $@"

# Compile debug version
$(DEBUG_TARGET): $(DEBUG_OBJECTS)
	@echo "Linking $@ (debug)..."
	$(NVCC) $(NVCC_DEBUG_FLAGS) $^ -o $@
	@echo "Debug build complete: $@"

# Compile object files (release)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile object files (debug)
$(OBJ_DIR)/%_debug.o: $(SRC_DIR)/%.cu
	@echo "Compiling $< (debug)..."
	$(NVCC) $(NVCC_DEBUG_FLAGS) -c $< -o $@

# Run the program
.PHONY: run
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Run with different stream counts
.PHONY: run-streams
run-streams: $(TARGET)
	@echo "Testing with different stream configurations..."
	@echo "\n=== 1 Stream ==="
	@./$(TARGET) 1
	@echo "\n=== 2 Streams ==="
	@./$(TARGET) 2
	@echo "\n=== 4 Streams ==="
	@./$(TARGET) 4
	@echo "\n=== 8 Streams ==="
	@./$(TARGET) 8

# Profile with nvprof (for older GPUs)
.PHONY: profile
profile: $(TARGET)
	@echo "Profiling with nvprof..."
	nvprof --print-gpu-trace ./$(TARGET)

# Profile with Nsight Compute (for newer GPUs)
.PHONY: profile-ncu
profile-ncu: $(TARGET)
	@echo "Profiling with Nsight Compute..."
	ncu --set full --export profile_report ./$(TARGET)

# Profile with Nsight Systems
.PHONY: profile-nsys
profile-nsys: $(TARGET)
	@echo "Profiling with Nsight Systems..."
	nsys profile -o profile_report ./$(TARGET)

# Check CUDA device info
.PHONY: deviceQuery
deviceQuery:
	@echo "CUDA Device Information:"
	@nvidia-smi

# Clean build files
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJ_DIR)
	@rm -rf $(BIN_DIR)
	@rm -f *.o
	@rm -f profile_report.*
	@echo "Clean complete"

# Clean and rebuild
.PHONY: rebuild
rebuild: clean all

# Show help
.PHONY: help
help:
	@echo "CUDA Optimized 2D Solver - Makefile Help"
	@echo "========================================"
	@echo "Targets:"
	@echo "  make              - Build release version"
	@echo "  make debug        - Build debug version with symbols"
	@echo "  make run          - Build and run the program"
	@echo "  make run-streams  - Test with different stream counts"
	@echo "  make profile      - Profile with nvprof"
	@echo "  make profile-ncu  - Profile with Nsight Compute"
	@echo "  make profile-nsys - Profile with Nsight Systems"
	@echo "  make deviceQuery  - Show CUDA device information"
	@echo "  make clean        - Remove build files"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH         - Target GPU architecture (current: $(CUDA_ARCH))"
	@echo ""
	@echo "Example: make CUDA_ARCH=sm_80"

# Detect CUDA architecture automatically (optional)
.PHONY: detect-arch
detect-arch:
	@echo "Detecting CUDA architecture..."
	@nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | awk '{printf "Detected: sm_%d%d\n", $$1*10, $$1*10%10}'

# Install target (optional)
.PHONY: install
install: $(TARGET)
	@echo "Installing to /usr/local/bin..."
	@sudo cp $(TARGET) /usr/local/bin/cuda_solver
	@echo "Installation complete"

# Uninstall target (optional)
.PHONY: uninstall
uninstall:
	@echo "Uninstalling from /usr/local/bin..."
	@sudo rm -f /usr/local/bin/cuda_solver
	@echo "Uninstall complete"

# Test target for different matrix sizes
.PHONY: test-sizes
test-sizes: $(TARGET)
	@echo "Testing different matrix sizes..."
	@echo "\n=== Small (1000x1000) ==="
	@./$(TARGET) 4 1000 1000
	@echo "\n=== Medium (3000x3000) ==="
	@./$(TARGET) 4 3000 3000
	@echo "\n=== Large (6000x6000) ==="
	@./$(TARGET) 4 6000 6000

# Show compiler info
.PHONY: info
info:
	@echo "CUDA Compiler Info:"
	@$(NVCC) --version
	@echo ""
	@echo "GPU Info:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

.PHONY: visualize
visualize: run
	python visualize_results.py 100 100

.PHONY: visualize-full
visualize-full: run
	python visualize_results.py 6000 6050 solution.bin
