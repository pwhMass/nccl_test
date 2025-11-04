# NCCL Test Makefile for Blue RDMA Driver (Mock Mode)

# Compiler settings
CXX = mpicxx
NVCC = nvcc

# Auto-detect CUDA installation
ifeq ($(CUDA_HOME),)
    ifneq ($(wildcard /usr/local/cuda-13.0),)
        CUDA_HOME := /usr/local/cuda-13.0
    else ifneq ($(wildcard /usr/local/cuda),)
        CUDA_HOME := /usr/local/cuda
    else ifneq ($(wildcard /usr/local/cuda-12.0),)
        CUDA_HOME := /usr/local/cuda-12.0
    else
        $(error CUDA not found. Please install CUDA or set CUDA_HOME)
    endif
endif

# Auto-detect NCCL installation (system-wide or custom)
ifeq ($(NCCL_HOME),)
    ifneq ($(wildcard /usr/include/nccl.h),)
        # NCCL installed system-wide via apt
        NCCL_INCLUDE := /usr/include
        NCCL_LIB := /usr/lib/x86_64-linux-gnu
    else ifneq ($(wildcard /usr/local/nccl),)
        NCCL_HOME := /usr/local/nccl
        NCCL_INCLUDE := $(NCCL_HOME)/include
        NCCL_LIB := $(NCCL_HOME)/lib
    else
        $(warning NCCL not found. Please install NCCL.)
    endif
else
    NCCL_INCLUDE := $(NCCL_HOME)/include
    NCCL_LIB := $(NCCL_HOME)/lib
endif

# MPI paths (OpenMPI on Ubuntu)
MPI_HOME ?= /usr/lib/x86_64-linux-gnu/openmpi

# Blue RDMA driver paths
BLUE_RDMA_ROOT = ../blue-rdma-driver
BLUE_RDMA_IBVERBS = $(BLUE_RDMA_ROOT)/dtld-ibverbs
RDMA_CORE_BUILD = $(BLUE_RDMA_IBVERBS)/rdma-core-55.0/build

# Compiler flags
CXXFLAGS = -std=c++11 -O2
NVCCFLAGS = -std=c++11 -O2

# Include paths
INCLUDES = -I$(CUDA_HOME)/include \
           -I$(RDMA_CORE_BUILD)/include

# Add NCCL include if available
ifdef NCCL_INCLUDE
    INCLUDES += -I$(NCCL_INCLUDE)
endif

# MPI includes are handled by mpicxx

# Library paths
LDFLAGS = -L$(CUDA_HOME)/lib64 \
          -L$(RDMA_CORE_BUILD)/lib \
          -L$(BLUE_RDMA_IBVERBS)/target/debug

# Add NCCL lib path if available
ifdef NCCL_LIB
    LDFLAGS += -L$(NCCL_LIB)
endif

# Libraries to link
LIBS = -lcudart -lnccl -lmpi -libverbs -lpthread -ldl

# Targets
TARGET = nccl_test
SINGLE_GPU_TARGET = single_gpu_test
TWO_PROCESS_TARGET = two_process_test
SOURCES = example.cpp
SINGLE_GPU_SOURCES = single_gpu_test.cpp
TWO_PROCESS_SOURCES = two_process_test.cpp

# Mock mode flag
MOCK ?= 1
ifeq ($(MOCK),1)
    CXXFLAGS += -DMOCK_MODE
    MOCK_FEATURES = --features mock
else
    MOCK_FEATURES =
endif

.PHONY: all clean rebuild driver single two_process

all: driver $(TARGET)

# Build all test programs
all_tests: driver $(TARGET) $(SINGLE_GPU_TARGET) $(TWO_PROCESS_TARGET)

# Build the Blue RDMA driver first
driver:
	@echo "Building Blue RDMA driver with mock mode..."
	cd $(BLUE_RDMA_IBVERBS) && cargo build --no-default-features $(MOCK_FEATURES)
	@if [ ! -f $(RDMA_CORE_BUILD)/lib/libibverbs.so.1 ]; then \
		echo "Building rdma-core..."; \
		cd $(RDMA_CORE_BUILD)/.. && ./build.sh; \
	fi

# Compile the test program (MPI-based)
$(TARGET): $(SOURCES)
	@echo "Compiling NCCL test program..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LDFLAGS) $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile single GPU test (no MPI)
$(SINGLE_GPU_TARGET): $(SINGLE_GPU_SOURCES)
	@echo "Compiling single GPU test..."
	nvcc $(CXXFLAGS) $(INCLUDES) -o $(SINGLE_GPU_TARGET) $(SINGLE_GPU_SOURCES) $(LDFLAGS) -lcudart -lnccl -libverbs -lpthread -ldl
	@echo "Build complete: $(SINGLE_GPU_TARGET)"

# Compile two-process test (no MPI, uses sockets)
$(TWO_PROCESS_TARGET): $(TWO_PROCESS_SOURCES)
	@echo "Compiling two-process test..."
	nvcc $(CXXFLAGS) $(INCLUDES) -o $(TWO_PROCESS_TARGET) $(TWO_PROCESS_SOURCES) $(LDFLAGS) -lcudart -lnccl -libverbs -lpthread -ldl
	@echo "Build complete: $(TWO_PROCESS_TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(SINGLE_GPU_TARGET) $(TWO_PROCESS_TARGET) *.o
	@echo "Clean complete"

# Rebuild everything
rebuild: clean all

# Run the test (single node, 2 processes) - DEPRECATED: use run_two instead
run: $(TARGET)
	@echo "⚠️  WARNING: This test doesn't work with single GPU (NCCL limitation)"
	@echo "   Use 'make run_single' for basic testing"
	@echo "   Or 'make run_two' for full network testing"
	@echo ""
	@echo "Running NCCL test with Blue RDMA driver (mock mode)..."
	@echo "Setting up environment..."
	export LD_LIBRARY_PATH=$(BLUE_RDMA_IBVERBS)/target/debug:$(RDMA_CORE_BUILD)/lib:$(CUDA_HOME)/lib64:$(NCCL_HOME)/lib:$$LD_LIBRARY_PATH && \
	export RUST_LOG=debug && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_NET=IB && \
	export NCCL_IB_HCA=bluerdma && \
	export NCCL_DEBUG=INFO && \
	export NCCL_DEBUG_SUBSYS=INIT,NET && \
	mpirun -np 2 --allow-run-as-root ./$(TARGET)

# Run single GPU test (validates driver loading)
run_single: $(SINGLE_GPU_TARGET)
	@echo "=========================================="
	@echo "  Running Single GPU Test"
	@echo "=========================================="
	@echo "This test validates:"
	@echo "  ✓ CUDA runtime"
	@echo "  ✓ NCCL initialization"
	@echo "  ✓ Blue RDMA driver loading"
	@echo "  ✓ Device query operations"
	@echo ""
	@echo "Note: No actual network communication (single process)"
	@echo ""
	export LD_LIBRARY_PATH=$(BLUE_RDMA_IBVERBS)/target/debug:$(RDMA_CORE_BUILD)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH && \
	export RUST_LOG=debug && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_NET=IB && \
	export NCCL_IB_HCA=bluerdma && \
	export NCCL_DEBUG=INFO && \
	export NCCL_DEBUG_SUBSYS=INIT,NET && \
	./$(SINGLE_GPU_TARGET)

# Run two-process test (full network testing)
run_two: $(TWO_PROCESS_TARGET)
	@echo "=========================================="
	@echo "  Running Two-Process Test"
	@echo "=========================================="
	@echo "This test requires TWO terminals:"
	@echo ""
	@echo "  Terminal 1 (Server):"
	@echo "    $$ make run_two_server"
	@echo ""
	@echo "  Terminal 2 (Client):"
	@echo "    $$ make run_two_client"
	@echo ""
	@echo "This test validates:"
	@echo "  ✓ All features from single GPU test"
	@echo "  ✓ Network communication between processes"
	@echo "  ✓ RDMA send/recv operations"
	@echo "  ✓ Completion queue polling"
	@echo ""

# Run two-process test - server (rank 0)
run_two_server: $(TWO_PROCESS_TARGET)
	@echo "Starting server (rank 0)..."
	@echo "Waiting for client connection..."
	@echo "⚠️  Note: Using NCCL_IGNORE_DISABLED_P2P=1 to allow single GPU testing"
	export LD_LIBRARY_PATH=$(BLUE_RDMA_IBVERBS)/target/debug:$(RDMA_CORE_BUILD)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH && \
	export RUST_LOG=debug && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_NET=IB && \
	export NCCL_IGNORE_DISABLED_P2P=1 && \
	export NCCL_P2P_DISABLE=1 && \
	export NCCL_DEBUG=INFO && \
	export NCCL_DEBUG_SUBSYS=INIT,NET && \
	export NCCL_DEBUG=WARN && \
	./$(TWO_PROCESS_TARGET) 0

# Run two-process test - client (rank 1)
run_two_client: $(TWO_PROCESS_TARGET)
	@echo "Starting client (rank 1)..."
	@echo "Connecting to server..."
	@echo "⚠️  Note: Using NCCL_IGNORE_DISABLED_P2P=1 to allow single GPU testing"
	export LD_LIBRARY_PATH=$(BLUE_RDMA_IBVERBS)/target/debug:$(RDMA_CORE_BUILD)/lib:$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH && \
	export RUST_LOG=debug && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_NET=IB && \
	export NCCL_IGNORE_DISABLED_P2P=1 && \
	export NCCL_P2P_DISABLE=1 && \
	export NCCL_DEBUG=INFO && \
	export NCCL_DEBUG_SUBSYS=INIT,NET && \
	./$(TWO_PROCESS_TARGET) 1

# Show configuration
info:
	@echo "==================================="
	@echo "  Build Configuration"
	@echo "==================================="
	@echo "CUDA_HOME:     $(CUDA_HOME)"
ifdef NCCL_INCLUDE
	@echo "NCCL_INCLUDE:  $(NCCL_INCLUDE)"
	@echo "NCCL_LIB:      $(NCCL_LIB)"
else
	@echo "NCCL:          Not found"
endif
	@echo "MPI_HOME:      $(MPI_HOME)"
	@echo "RDMA Core:     $(RDMA_CORE_BUILD)"
	@echo ""
	@echo "Compiler:      $(CXX)"
	@echo "Flags:         $(CXXFLAGS)"
	@echo "Includes:      $(INCLUDES)"
	@echo "Libs:          $(LIBS)"
	@echo ""

# Help target
help:
	@echo "================================================"
	@echo "  NCCL Blue RDMA Driver Test Suite"
	@echo "================================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all          - Build driver and MPI test (default)"
	@echo "  all_tests    - Build all test programs"
	@echo "  driver       - Build Blue RDMA driver only"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild"
	@echo ""
	@echo "Test Targets:"
	@echo "  run_single       - Run single GPU test (driver validation)"
	@echo "  run_two_server   - Run two-process test server (rank 0)"
	@echo "  run_two_client   - Run two-process test client (rank 1)"
	@echo "  run              - (Deprecated) MPI test - doesn't work with 1 GPU"
	@echo ""
	@echo "Info Targets:"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Options:"
	@echo "  MOCK=1       - Build with mock mode support (default)"
	@echo ""
	@echo "Recommended Testing Flow:"
	@echo "  1. make run_single      # Validate driver loading"
	@echo "  2. Terminal 1: make run_two_server"
	@echo "     Terminal 2: make run_two_client"
	@echo ""
	@echo ""
	@echo "Example usage:"
	@echo "  make info          # Show detected paths"
	@echo "  make MOCK=1        # Build with mock mode"
	@echo "  make run MOCK=1    # Build and run with mock mode"
