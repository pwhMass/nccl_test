# RCCL Test Makefile for ROCm and Blue RDMA

# Compiler settings
CXX = mpicxx
HIPCC = hipcc

# ROCm paths
ROCM_HOME ?= /opt/rocm
DTK_HOME ?= /opt/dtk

# RCCL paths
RCCL_INCLUDE ?= $(DTK_HOME)/include
RCCL_LIB ?= $(DTK_HOME)/lib

# MPI paths
MPI_HOME = /usr/mpi/gcc/openmpi-4.1.7rc1
MPI_INCLUDE = $(MPI_HOME)/include
MPI_LIB = $(MPI_HOME)/lib

# Compiler flags
CXXFLAGS = -std=c++11 -O2
HIPCCFLAGS = -std=c++11 -O2

# Include paths
INCLUDES = -I$(RCCL_INCLUDE) \
           -I$(MPI_INCLUDE) \
           -I$(ROCM_HOME)/include

# Library paths
LDFLAGS = -L$(RCCL_LIB) \
          -L$(MPI_LIB) \
          -L$(ROCM_HOME)/lib

# Libraries to link
LIBS = -lrccl -lmpi

# Targets
TARGET = test
SOURCES = simple_test.cpp
NORMAL_TARGET = normal_test
NORMAL_SOURCES = normal_test.cpp
NOMPI_TARGET = normal_test_nompi
NOMPI_SOURCES = normal_test_nompi.cpp

.PHONY: all clean rebuild run run_rdma run_rdma_force run_tcp normal normal_rdma normal_rdma_force normal_tcp normal_nompi_rank0 normal_nompi_rank1 info help

all: $(TARGET) $(NORMAL_TARGET) $(NOMPI_TARGET)

# Compile the simple test program
$(TARGET): $(SOURCES)
	@echo "Compiling RCCL simple test program..."
	$(HIPCC) $(HIPCCFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LDFLAGS) $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile the normal test program (with AllReduce)
$(NORMAL_TARGET): $(NORMAL_SOURCES)
	@echo "Compiling RCCL normal test program (AllReduce)..."
	$(HIPCC) $(HIPCCFLAGS) $(INCLUDES) -o $(NORMAL_TARGET) $(NORMAL_SOURCES) $(LDFLAGS) $(LIBS)
	@echo "Build complete: $(NORMAL_TARGET)"

# Compile the normal test program without MPI
$(NOMPI_TARGET): $(NOMPI_SOURCES)
	@echo "Compiling RCCL normal test program without MPI..."
	$(HIPCC) $(HIPCCFLAGS) -I$(RCCL_INCLUDE) -I$(ROCM_HOME)/include -o $(NOMPI_TARGET) $(NOMPI_SOURCES) -L$(RCCL_LIB) -L$(ROCM_HOME)/lib -lrccl
	@echo "Build complete: $(NOMPI_TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET) $(NORMAL_TARGET) $(NOMPI_TARGET) *.o
	@echo "Clean complete"

# Rebuild everything
rebuild: clean all

# Run with RDMA (2 DCUs) - Each rank uses different RDMA device
run_rdma: $(TARGET)
	@echo "=========================================="
	@echo "  Running RCCL Test with RDMA (2 DCUs)"
	@echo "=========================================="
	@echo "Rank 0 -> blue0, Rank 1 -> blue1"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_NET_GDR_LEVEL=0 && \
	mpirun -np 2 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_IB_DISABLE=0 \
	  -x RUST_LOG=debug \
	  -mca coll ^hcoll \
	  ./$(TARGET)

# Run with RDMA forcing network transport (disables P2P and SHM for testing)
run_rdma_force: $(TARGET)
	@echo "=========================================="
	@echo "  Running RCCL Test - Force IB Network"
	@echo "=========================================="
	@echo "Disabling P2P and SHM to force IB network usage"
	@echo "Using bluerdma0 and bluerdma1"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_P2P_DISABLE=1 && \
	export NCCL_SHM_DISABLE=1 && \
	export NCCL_NET_GDR_LEVEL=0 && \
	export NCCL_IB_HCA=bluerdma0,bluerdma1 && \
	mpirun -np 2 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_DEBUG_SUBSYS=INIT,NET \
	  -x NCCL_IB_DISABLE=0 \
	  -x NCCL_IB_HCA=bluerdma0,bluerdma1 \
	  -x NCCL_P2P_DISABLE=1 \
	  -x NCCL_SHM_DISABLE=1 \
	  -x NCCL_NET_GDR_LEVEL=0 \
	  -x RUST_LOG=debug \
	  -mca coll ^hcoll \
	  ./$(TARGET)

# Run with RDMA (4 DCUs) - Uses both RDMA devices
run_rdma_4: $(TARGET)
	@echo "=========================================="
	@echo "  Running RCCL Test with RDMA (4 DCUs)"
	@echo "=========================================="
	@echo "Using blue0 and blue1 RDMA devices"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	mpirun -np 4 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_IB_DISABLE=0 \
	  -mca coll ^hcoll \
	  ./$(TARGET)

# Run with TCP (2 DCUs)
run_tcp: $(TARGET)
	@echo "=========================================="
	@echo "  Running RCCL Test with TCP (2 DCUs)"
	@echo "=========================================="
	@echo "Using TCP instead of RDMA"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	mpirun -np 2 \
	  -x UCX_TLS=tcp,self \
	  -x HCOLL_ENABLE_MCAST=0 \
	  -x NCCL_DEBUG=INFO \
	  ./$(TARGET)

# Run with TCP (4 DCUs)
run_tcp_4: $(TARGET)
	@echo "=========================================="
	@echo "  Running RCCL Test with TCP (4 DCUs)"
	@echo "=========================================="
	@echo "Using TCP instead of RDMA"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	mpirun -np 4 \
	  -x UCX_TLS=tcp,self \
	  -x HCOLL_ENABLE_MCAST=0 \
	  -x NCCL_DEBUG=INFO \
	  ./$(TARGET)

# Run normal_test with RDMA (2 DCUs) - Tests actual data transfer
normal_rdma: $(NORMAL_TARGET)
	@echo "=========================================="
	@echo "  Running Normal Test with RDMA (2 DCUs)"
	@echo "=========================================="
	@echo "Testing AllReduce with RDMA"
	@echo "Rank 0 -> blue0, Rank 1 -> blue1"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	mpirun -np 2 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_IB_DISABLE=0 \
	  -mca coll ^hcoll \
	  ./$(NORMAL_TARGET)

# Run normal_test forcing IB network (disables P2P and SHM)
normal_rdma_force: $(NORMAL_TARGET)
	@echo "=========================================="
	@echo "  Normal Test - Force IB Network (2 DCUs)"
	@echo "=========================================="
	@echo "Disabling P2P and SHM to force IB network usage"
	@echo "Using bluerdma0 and bluerdma1"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_P2P_DISABLE=1 && \
	export NCCL_SHM_DISABLE=1 && \
	export NCCL_NET_GDR_LEVEL=0 && \
	export NCCL_IB_HCA=bluerdma0,bluerdma1 && \
	mpirun -np 2 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_DEBUG_SUBSYS=INIT,NET \
	  -x NCCL_IB_DISABLE=0 \
	  -x NCCL_IB_HCA=bluerdma0,bluerdma1 \
	  -x NCCL_P2P_DISABLE=1 \
	  -x NCCL_SHM_DISABLE=1 \
	  -x NCCL_NET_GDR_LEVEL=0 \
	  -x RUST_LOG=debug \
	  -mca coll ^hcoll \
	  ./$(NORMAL_TARGET)

# Run normal_test with RDMA (4 DCUs)
normal_rdma_4: $(NORMAL_TARGET)
	@echo "=========================================="
	@echo "  Running Normal Test with RDMA (4 DCUs)"
	@echo "=========================================="
	@echo "Testing AllReduce with RDMA"
	@echo "Using blue0 and blue1 RDMA devices"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	mpirun -np 4 \
	  -x UCX_NET_DEVICES=blue0,blue1 \
	  -x NCCL_DEBUG=INFO \
	  -x NCCL_IB_DISABLE=0 \
	  -mca coll ^hcoll \
	  ./$(NORMAL_TARGET)

# Run normal_test with TCP (2 DCUs)
normal_tcp: $(NORMAL_TARGET)
	@echo "=========================================="
	@echo "  Running Normal Test with TCP (2 DCUs)"
	@echo "=========================================="
	@echo "Testing AllReduce with TCP"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	mpirun -np 2 \
	  -x UCX_TLS=tcp,self \
	  -x HCOLL_ENABLE_MCAST=0 \
	  -x NCCL_DEBUG=INFO \
	  ./$(NORMAL_TARGET)

# Run normal_test with TCP (4 DCUs)
normal_tcp_4: $(NORMAL_TARGET)
	@echo "=========================================="
	@echo "  Running Normal Test with TCP (4 DCUs)"
	@echo "=========================================="
	@echo "Testing AllReduce with TCP"
	@echo ""
	export PATH=$(MPI_HOME)/bin:$$PATH && \
	export LD_LIBRARY_PATH=$(MPI_LIB):$$LD_LIBRARY_PATH && \
	export NCCL_DEBUG=INFO && \
	mpirun -np 4 \
	  -x UCX_TLS=tcp,self \
	  -x HCOLL_ENABLE_MCAST=0 \
	  -x NCCL_DEBUG=INFO \
	  ./$(NORMAL_TARGET)

# Run normal_test_nompi as rank 0 (server) with blue0 RDMA device
normal_nompi_rank0: $(NOMPI_TARGET)
	@echo "=========================================="
	@echo "  Normal Test No-MPI - Rank 0 (Server)"
	@echo "=========================================="
	@echo "Using RDMA device: bluerdma0 only"
	@echo "Run 'make normal_nompi_rank1' in another terminal"
	@echo ""
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_IB_HCA=bluerdma0 && \
	export UCX_NET_DEVICES=blue0 && \
	export NCCL_NET_GDR_LEVEL=0 && \
		export NCCL_P2P_DISABLE=1 && \
	export NCCL_SHM_DISABLE=1 && \
	./$(NOMPI_TARGET) 0

# Run normal_test_nompi as rank 1 (client) with blue1 RDMA device
normal_nompi_rank1: $(NOMPI_TARGET)
	@echo "=========================================="
	@echo "  Normal Test No-MPI - Rank 1 (Client)"
	@echo "=========================================="
	@echo "Using RDMA device: bluerdma1 only"
	@echo "Connecting to rank 0..."
	@echo ""
	export NCCL_DEBUG=INFO && \
	export NCCL_IB_DISABLE=0 && \
	export NCCL_IB_HCA=bluerdma1 && \
	export UCX_NET_DEVICES=blue1 && \
	export NCCL_NET_GDR_LEVEL=0 && \
		export NCCL_P2P_DISABLE=1 && \
	export NCCL_SHM_DISABLE=1 && \
	./$(NOMPI_TARGET) 1

# Default run target (RDMA with 2 DCUs)
run: run_rdma

# Default normal test target
normal: normal_rdma

# Show configuration
info:
	@echo "==================================="
	@echo "  Build Configuration"
	@echo "==================================="
	@echo "ROCM_HOME:     $(ROCM_HOME)"
	@echo "DTK_HOME:      $(DTK_HOME)"
	@echo "RCCL_INCLUDE:  $(RCCL_INCLUDE)"
	@echo "RCCL_LIB:      $(RCCL_LIB)"
	@echo "MPI_HOME:      $(MPI_HOME)"
	@echo "MPI_INCLUDE:   $(MPI_INCLUDE)"
	@echo "MPI_LIB:       $(MPI_LIB)"
	@echo ""
	@echo "Compiler:      $(HIPCC)"
	@echo "Flags:         $(HIPCCFLAGS)"
	@echo "Includes:      $(INCLUDES)"
	@echo "Libs:          $(LIBS)"
	@echo ""
	@echo "RDMA Devices:"
	@rdma link show 2>/dev/null || echo "  No RDMA devices found"
	@echo ""
	@echo "DCU Devices:"
	@rocm-smi --showid 2>/dev/null || echo "  No DCU devices found"
	@echo ""

# Help target
help:
	@echo "================================================"
	@echo "  RCCL Test Suite for ROCm and Blue RDMA"
	@echo "================================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all          - Build RCCL test program (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  rebuild      - Clean and rebuild"
	@echo ""
	@echo "Run Targets (Simple Test):"
	@echo "  run              - Run simple test with RDMA (2 DCUs) - default"
	@echo "  run_rdma         - Run simple test with RDMA (2 DCUs)"
	@echo "  run_rdma_force   - Force IB network (disable P2P)"
	@echo "  run_rdma_4       - Run simple test with RDMA (4 DCUs)"
	@echo "  run_tcp          - Run simple test with TCP (2 DCUs)"
	@echo "  run_tcp_4        - Run simple test with TCP (4 DCUs)"
	@echo ""
	@echo "Run Targets (Normal Test - AllReduce):"
	@echo "  normal              - Run normal test with RDMA (2 DCUs) - default"
	@echo "  normal_rdma         - Run normal test with RDMA (2 DCUs)"
	@echo "  normal_rdma_force   - Force IB network (disable P2P)"
	@echo "  normal_rdma_4       - Run normal test with RDMA (4 DCUs)"
	@echo "  normal_tcp          - Run normal test with TCP (2 DCUs)"
	@echo "  normal_tcp_4        - Run normal test with TCP (4 DCUs)"
	@echo ""
	@echo "Run Targets (Normal Test - No MPI):"
	@echo "  normal_nompi_rank0  - Run rank 0 with blue0 (run in terminal 1)"
	@echo "  normal_nompi_rank1  - Run rank 1 with blue1 (run in terminal 2)"
	@echo ""
	@echo "Info Targets:"
	@echo "  info         - Show build configuration"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all programs"
	@echo "  make info               # Show configuration"
	@echo "  make run_rdma           # Run simple test with RDMA (2 DCUs)"
	@echo "  make normal_rdma        # Run AllReduce test with RDMA (2 DCUs)"
	@echo "  make normal_tcp         # Run AllReduce test with TCP (2 DCUs)"
	@echo ""
	@echo "  # Run without MPI (in two terminals):"
	@echo "  Terminal 1: make normal_nompi_rank0"
	@echo "  Terminal 2: make normal_nompi_rank1"
	@echo ""
	@echo "Environment Variables (optional):"
	@echo "  NCCL_DEBUG=INFO     - Enable RCCL debug output"
	@echo "  NCCL_IB_DISABLE=0   - Enable InfiniBand/RDMA"
	@echo "  UCX_NET_DEVICES     - Specify network device"
	@echo ""
