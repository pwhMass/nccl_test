# NCCL Test for Blue RDMA Driver

This directory contains a NCCL (NVIDIA Collective Communications Library) test program designed to test the Blue RDMA driver in mock mode using IB (InfiniBand) transport.

## Quick Start (WSL/Ubuntu 24.04)

```bash
# 1. Check your environment
./check_env.sh

# 2. Install dependencies (NCCL + MPI)
./install_nccl_commands.sh

# 3. Build the test
make MOCK=1

# 4. Run the test
./run_test.sh
```

---

## Installation Process (Actual Setup Documentation)

This section documents the **complete installation process** performed on our WSL2 environment, including all issues encountered and their solutions.

### System Environment

**Configuration:**
- **OS**: Ubuntu 24.04.3 LTS (WSL2 on Windows)
- **GPU**: NVIDIA GeForce RTX 4060 Ti (16GB)
- **Windows Driver**: 576.02 (Host)
- **CUDA Toolkit**: 13.0.96 (WSL Linux: `/usr/local/cuda-13.0`)
- **CUDA Driver Support**: 12.9 (Reported by nvidia-smi)
- **WSL Kernel**: 6.6.87.2-microsoft-standard-WSL2

### Complete Installation Steps

#### Step 1: Install OpenMPI

```bash
sudo apt update
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
```

**Verify Installation:**
```bash
mpirun --version
# Output: mpirun (Open MPI) 4.1.x
mpicxx --version
```

#### Step 2: Install NCCL from Ubuntu Repository

**Critical Information:**
- Ubuntu 24.04 provides **NCCL 2.18.5** in the official `multiverse` repository
- This version supports **CUDA 11.x through 13.x**
- ⚠️ **DO NOT** attempt to install NCCL 2.28.7 (requires CUDA 12.x from different sources)

**Installation:**
```bash
sudo apt update
sudo apt install -y libnccl2 libnccl-dev
```

**Verify Installation:**
```bash
# Check installed packages
dpkg -l | grep nccl
# Expected:
# ii  libnccl2      2.18.5-1-2  amd64  NVIDIA Optimized primitives
# ii  libnccl-dev   2.18.5-1-2  amd64  Development files

# Check files
ls /usr/include/nccl.h
ls /usr/lib/x86_64-linux-gnu/libnccl.so.2

# Check version from header
grep "NCCL_VERSION" /usr/include/nccl.h
```

#### Step 3: Verify CUDA Installation

```bash
# Check CUDA toolkit
/usr/local/cuda-13.0/bin/nvcc --version
# Output: Cuda compilation tools, release 13.0, V13.0.88

# Check GPU driver
nvidia-smi
# Look for: Driver Version: 576.02, CUDA Version: 12.9

# Add CUDA to environment (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

#### Step 4: Build Blue RDMA Driver (Mock Mode)

```bash
cd ../dtld-ibverbs
cargo build --no-default-features --features mock

# Build rdma-core if not already built
cd rdma-core-55.0
./build.sh
cd ../..
```

**Verify Driver Build:**
```bash
ls dtld-ibverbs/target/debug/libbluerdma_rust.so
ls dtld-ibverbs/rdma-core-55.0/build/lib/libibverbs.so.1
ls dtld-ibverbs/rdma-core-55.0/build/lib/libbluerdma-rdmav34.so
```

#### Step 5: Build NCCL Test Program

```bash
cd nccl_test

# Check build configuration
make info
# Should show:
#   CUDA_HOME: /usr/local/cuda-13.0
#   NCCL_INCLUDE: /usr/include
#   NCCL_LIB: /usr/lib/x86_64-linux-gnu

# Clean and build
make clean
make MOCK=1
```

**Expected Build Output:**
```
Building Blue RDMA driver with mock mode...
Compiling NCCL test program...
mpicxx -std=c++11 -O2 -Wall -I/usr/local/cuda-13.0/include ...
Build complete: nccl_test
```

#### Step 6: Run the Test

```bash
./run_test.sh
```

---

### Known Issues and Solutions

#### ❌ Issue 1: CUDA Driver Version Mismatch

**Symptom:**
```
Failed: Cuda error example.cpp:59 'CUDA driver version is insufficient for CUDA runtime version'
```

**Root Cause Analysis:**

In WSL2, the GPU driver runs on the **Windows host**, not in Linux:

```
Windows Host (Driver 576.02)
    ↓ reports max CUDA support: 12.9
WSL2 Linux
    ↓ has CUDA Toolkit: 13.0.96
Application tries cudaSetDevice()
    ↓ CUDA Runtime 13.0 requires driver >= 530.x with CUDA 13.0 support
    ✗ ERROR: Driver insufficient
```

**Version Requirements:**
| CUDA Toolkit | Minimum Driver Version | Notes |
|--------------|------------------------|-------|
| 13.0.x | 530.30+ | Requires explicit CUDA 13.0 support |
| 12.9.x | 525.60+ | Our driver supports this |
| 12.6.x | 525.60+ | Also compatible |

**Solution Option 1: Update Windows GPU Driver (Recommended)**

1. **Download Latest Driver:**
   - Visit: https://www.nvidia.com/Download/index.aspx
   - Select: GeForce RTX 4060 Ti, Windows version
   - Download Game Ready or Studio Driver (latest)

2. **Install on Windows Host:**
   - Run installer as Administrator
   - Select "Custom Installation" → "Clean Installation" if needed

3. **Restart WSL2:**
   ```powershell
   # In Windows PowerShell
   wsl --shutdown
   wsl
   ```

4. **Verify in WSL:**
   ```bash
   nvidia-smi
   # Check: CUDA Version should be >= 13.0
   ```

5. **Rerun Test:**
   ```bash
   cd /home/peng/projects/rdma_all/nccl_test
   ./run_test.sh
   ```

**Solution Option 2: Downgrade CUDA Toolkit to 12.x**

If updating the Windows driver is not possible:

```bash
# Option A: Install CUDA 12.6 from NVIDIA repo
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6

# Update symlink
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-12.6 /usr/local/cuda

# Rebuild test
cd /home/peng/projects/rdma_all/nccl_test
make clean
make MOCK=1
```

#### ❌ Issue 2: NCCL Version Not Found (2.28.7)

**Symptom:**
```bash
sudo apt install libnccl2=2.28.7-1+cuda13.0
# E: Version '2.28.7-1+cuda13.0' for 'libnccl2' was not found
```

**Root Cause:**
- NVIDIA's documentation references **NCCL 2.28.7** for **CUDA 12.x**
- Ubuntu 24.04 only provides **NCCL 2.18.5**
- The WSL CUDA repository does not include NCCL packages

**Understanding Repository Sources:**

```bash
# Check which repositories provide NCCL
apt-cache policy libnccl2
# Output shows:
#   Candidate: 2.18.5-1-2
#   500 http://archive.ubuntu.com/ubuntu noble/multiverse

# Your system has these repositories:
ls /etc/apt/sources.list.d/
# cuda-wsl-ubuntu-x86_64.list  ← CUDA toolkit only, NO NCCL
# ubuntu.sources               ← System packages, has NCCL 2.18.5
```

**Solution:**
Use Ubuntu's version (2.18.5), which is perfectly compatible:

```bash
sudo apt install libnccl2 libnccl-dev
# This installs 2.18.5-1-2, which supports CUDA 11.x - 13.x
```

**Why This Works:**
| NCCL Version | CUDA Support | Status |
|--------------|--------------|--------|
| 2.18.5 | 11.x - 13.x | ✅ Available in Ubuntu 24.04 |
| 2.28.7 | 12.x only | ❌ Not in Ubuntu repos, requires CUDA 12.x |

#### ⚠️ Issue 3: NCCL Plugin Warning

**Symptom:**
```
NCCL INFO NET/Plugin : Plugin load (libnccl-net.so) returned 2 :
libnccl-net.so: cannot open shared object file: No such file or directory
NCCL INFO NET/Plugin : No plugin found, using internal implementation
```

**Status:** **This is EXPECTED and can be ignored**

**Explanation:**
- NCCL first tries to load external network plugins (e.g., for AWS EFA, Azure IB)
- Plugin not found → Falls back to internal IB Verbs implementation
- Blue RDMA driver is accessed through IB Verbs API, not plugin system
- The fallback is the correct path for our use case

---

### Understanding the Architecture

#### Why Do We Need MPI?

**Component Responsibilities:**

```
┌─────────────────────────────────────────┐
│ Process 0 (Rank 0)   Process 1 (Rank 1) │
│        ↓                     ↓           │
│   ┌────────┐           ┌────────┐       │
│   │  MPI   │ ←─────→  │  MPI   │       │ Control Plane
│   └────────┘           └────────┘       │ • Launch processes
│        ↓                     ↓           │ • Exchange metadata
│   ┌────────┐           ┌────────┐       │ • Synchronization
│   │  NCCL  │ ←═══════→ │  NCCL  │       │ Data Plane
│   └────────┘  via RDMA  └────────┘      │ • GPU data transfer
│        ↓                     ↓           │ • Collective ops
│   ┌────────┐           ┌────────┐       │
│   │ Blue   │ ←─────→  │ Blue   │       │ Transport
│   │ RDMA   │           │ RDMA   │       │ • IB Verbs API
│   └────────┘           └────────┘       │ • Mock impl
└─────────────────────────────────────────┘
```

**From example.cpp:**
```cpp
// MPI: Bootstrap and coordinate
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get my rank
MPI_Comm_size(MPI_COMM_WORLD, &world_size);

// MPI: Exchange NCCL initialization data
if (rank == 0) {
    ncclGetUniqueId(&id);  // Root generates ID
}
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
// ↑ All processes now have the same NCCL ID

// NCCL: Initialize communicator using the ID
ncclCommInitRank(&comm, world_size, id, rank);

// NCCL: Actual GPU data transfer (via Blue RDMA)
ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream);
// ↑ This goes through IB Verbs → Blue RDMA driver
```

**Why Not Just Use MPI for Everything?**
- MPI is not GPU-aware by default
- NCCL has optimized GPU collective algorithms
- NCCL supports GPU Direct RDMA (bypass CPU memory)

#### APT Repository System

**How APT Works:**

```
sudo apt install libnccl2
         ↓
APT reads: /etc/apt/sources.list.d/*.list
         ↓
Queries cache: /var/lib/apt/lists/
         ↓
Finds packages from:
    1. http://archive.ubuntu.com/ubuntu (Ubuntu repo)
       └→ libnccl2: 2.18.5-1-2 ✅ Available
    2. https://developer.download.nvidia.com/.../wsl-ubuntu/
       └→ libnccl2: Not provided
         ↓
Selects: 2.18.5-1-2 from Ubuntu
         ↓
Downloads: .deb file from Ubuntu mirrors
         ↓
Installs to:
    /usr/lib/x86_64-linux-gnu/libnccl.so.2
    /usr/include/nccl.h
```

**Check Your Sources:**
```bash
# View repository configuration
ls /etc/apt/sources.list.d/

# See which repo provides a package
apt-cache policy libnccl2

# See all available versions
apt-cache madison libnccl2

# Search for packages
apt-cache search nccl
```

---

### Version Compatibility Reference

| Component | Installed Version | Requirements | Status |
|-----------|-------------------|--------------|--------|
| **Ubuntu** | 24.04.3 LTS | - | ✅ |
| **CUDA Toolkit** | 13.0.96 | Driver >= 530.x | ⚠️ Driver mismatch |
| **CUDA Driver (Win)** | 576.02 | Reports CUDA 12.9 | ⚠️ Update needed |
| **NCCL** | 2.18.5-1-2 | CUDA 11.x - 13.x | ✅ |
| **OpenMPI** | 4.1.x | - | ✅ |
| **Blue RDMA** | Mock mode | Rust 1.70+ | ✅ |

**CUDA Version Compatibility:**
```
CUDA 13.0 → requires driver 530.30+
CUDA 12.9 → requires driver 525.60+  ← Your driver supports this
CUDA 12.6 → requires driver 525.60+
```

---

## Overview

The test program performs an `AllReduce` operation using NCCL over your Blue RDMA driver, verifying that RDMA communication works correctly.

## Architecture

```
Application (nccl_test)
    ↓ NCCL API
NCCL Library (libnccl.so)
    ↓ IB Verbs Plugin
Blue RDMA Driver (libbluerdma_rust.so + libibverbs.so)
    ↓ Mock Mode
Software-based RDMA Emulation
```

## Prerequisites

1. **CUDA Toolkit** (for GPU operations)
   ```bash
   # Check if CUDA is installed
   nvcc --version
   ```

2. **NCCL Library** (for collective communication)

   **Automatic Installation (Recommended):**
   ```bash
   ./install_deps.sh  # Installs via network repository
   ```

   **Manual Installation:**

   Two methods available:

   - **Network Repository** (Recommended for WSL development):
     ```bash
     # Add NVIDIA repository
     wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
     sudo dpkg -i cuda-keyring_1.1-1_all.deb
     sudo apt update

     # Install NCCL
     sudo apt install libnccl2 libnccl-dev
     ```
     ✅ Pros: Quick, automatic updates, easy to maintain
     ⚠️ Note: May upgrade CUDA (use version locking if needed)

   - **Local Repository** (For offline or production):
     ```bash
     # Download from https://developer.nvidia.com/nccl/nccl-download
     sudo dpkg -i nccl-repo-<version>.deb
     sudo apt update
     sudo apt install libnccl2 libnccl-dev
     ```
     ✅ Pros: Offline capable, version locked
     ❌ Cons: Manual download required, harder to update

3. **MPI** (for multi-process coordination)
   ```bash
   # Install OpenMPI
   sudo apt-get install openmpi-bin libopenmpi-dev
   ```

4. **Blue RDMA Driver** (built with mock support)
   ```bash
   cd ../dtld-ibverbs
   cargo build --no-default-features --features mock
   cd rdma-core-55.0 && ./build.sh
   ```

## Building

### Quick Build
```bash
make MOCK=1
```

### Manual Build
```bash
# 1. Build the Blue RDMA driver
cd ../dtld-ibverbs
cargo build --no-default-features --features mock
cd rdma-core-55.0 && ./build.sh

# 2. Build the test program
cd ../../nccl_test
make
```

## Running the Test

### Using the Test Script (Recommended)
```bash
# Run with default settings (2 processes)
./run_test.sh

# Run with custom number of processes
./run_test.sh 4

# Run with custom data size
./run_test.sh 2 2048
```

### Manual Execution
```bash
# Set environment variables
export LD_LIBRARY_PATH=../dtld-ibverbs/target/debug:../dtld-ibverbs/rdma-core-55.0/build/lib:$LD_LIBRARY_PATH
export RUST_LOG=debug
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_IB_HCA=bluerdma
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET

# Run with MPI
mpirun -np 2 --allow-run-as-root ./nccl_test
```

## What the Test Does

1. **Initializes MPI** for multi-process communication
2. **Configures NCCL** to use IB transport with the Blue RDMA device
3. **Allocates GPU memory** for send and receive buffers
4. **Performs AllReduce** operation: each rank sends `rank+1`, result should be sum of all ranks
5. **Verifies results** by checking if all elements equal expected sum
6. **Cleans up** GPU memory and NCCL communicator

### Expected Output

For 2 processes, each rank should report:
```
[Rank 0] Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 1] Test PASSED: result[0] = 3.0 (expected 3.0)
```

The expected sum is calculated as: `(world_size * (world_size + 1)) / 2`
- For 2 processes: (2 * 3) / 2 = 3.0

## Environment Variables

### Blue RDMA Driver
- `RUST_LOG=debug` - Enable debug logging for the Rust driver
- `LD_LIBRARY_PATH` - Path to Blue RDMA libraries

### NCCL Configuration
- `NCCL_IB_DISABLE=0` - Enable IB transport
- `NCCL_NET=IB` - Use IB network backend
- `NCCL_IB_HCA=bluerdma` - Specify Blue RDMA device
- `NCCL_DEBUG=INFO` - NCCL debug level
- `NCCL_DEBUG_SUBSYS=INIT,NET` - Debug initialization and networking

## Troubleshooting

### NCCL doesn't detect Blue RDMA device
```bash
# Check if device is registered
export LD_LIBRARY_PATH=../dtld-ibverbs/target/debug:../dtld-ibverbs/rdma-core-55.0/build/lib
../dtld-ibverbs/rdma-core-55.0/build/bin/ibv_devices
```

### Build errors
```bash
# Clean and rebuild
make clean
make MOCK=1
```

### Runtime errors
```bash
# Enable verbose logging
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL
export RUST_LOG=trace

# Run the test again
./run_test.sh
```

## File Structure

```
nccl_test/
├── example.cpp      # Main test program
├── Makefile         # Build configuration
├── run_test.sh      # Test runner script
└── README.md        # This file
```

## Key Code Sections

### NCCL Configuration (example.cpp:41-45)
```cpp
setenv("NCCL_IB_DISABLE", "0", 1);           // Enable IB
setenv("NCCL_NET", "IB", 1);                 // Use IB network
setenv("NCCL_IB_HCA", "bluerdma", 1);       // Blue RDMA device
setenv("NCCL_DEBUG", "INFO", 1);             // Debug output
```

### AllReduce Operation (example.cpp:82)
```cpp
ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream);
```

## Notes

- This test runs in **mock mode** - no real hardware required
- The test uses **local communication** (all processes on same node)
- For multi-node testing, you'll need to configure MPI hostfiles
- GPU device selection is based on rank: `cudaSetDevice(rank % 2)`

## Next Steps

1. **Verify basic functionality** with this test
2. **Test different data sizes** and operation types
3. **Add more collective operations** (Broadcast, Reduce, Gather, etc.)
4. **Test multi-node scenarios** if you have multiple machines
5. **Profile performance** using NCCL's built-in profiling

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Blue RDMA Driver Architecture](../CLAUDE.md)




之前 12.0 nccl版本时出现问题：

[2025-11-03T17:53:42.591650400Z WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml, using default value
[2025-11-03T17:53:42.591690844Z DEBUG blue_rdma_driver::verbs::core] before open default
[2025-11-03T17:53:42.592425704Z INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[2025-11-03T17:53:42.592452425Z INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
[2025-11-03T17:53:42.592496079Z DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[2025-11-03T17:53:42.592518525Z DEBUG blue_rdma_driver::verbs::core] before load default
[2025-11-03T17:53:42.592529467Z WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml, using default value
[2025-11-03T17:53:42.592537693Z DEBUG blue_rdma_driver::verbs::core] before open default
[2025-11-03T17:53:42.593047463Z INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[2025-11-03T17:53:42.593084894Z INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes

[2025-11-04 01:53:42] DESKTOP-M211L3D:57639:57639 [0] transport/net_ib.cc:526 NCCL WARN Could not find real path of bluerdma1 (/sys/class/infiniband/bluerdma1/device)
DESKTOP-M211L3D:57639:57639 [0] NCCL INFO NET/IB: [1] bluerdma1:uverbs1:1/RoCE provider=None speed=400000 context=0x5c3144bb5160 pciPath=(null) ar=0
DESKTOP-M211L3D:57639:57639 [0] NCCL INFO NET/IB : Made virtual device [0] name=bluerdma1 speed=400000 ndevs=1
DESKTOP-M211L3D:57639:57639 [0] NCCL INFO NET/IB : Using [0]bluerdma1:1/RoCE [RO]; OOB eth0:100.82.177.43<0>
DESKTOP-M211L3D:57639:57639 [0] NCCL INFO Initialized NET plugin IB
Segmentation fault (core dumped)
make: *** [Makefile:213: run_two_client] Error 139

现在 13.0 nccl 版本时出现：

bluerdma device allocated
Setting op alloc_pd
Setting op create_cq
Setting op create_qp
Setting op dealloc_pd
Setting op dereg_mr
Setting op destroy_cq
Setting op destroy_qp
Setting op modify_qp
Setting op poll_cq
Setting op post_recv
Setting op post_send
Setting op query_device_ex
Setting op query_port
Setting op query_qp
Setting op reg_mr
bluerdma device allocated
[2025-11-03T17:57:24.093100592Z DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[2025-11-03T17:57:24.093196723Z DEBUG blue_rdma_driver::verbs::core] before load default
[2025-11-03T17:57:24.093240430Z WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml, using default value
[2025-11-03T17:57:24.093289810Z DEBUG blue_rdma_driver::verbs::core] before open default
[2025-11-03T17:57:24.093526571Z INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[2025-11-03T17:57:24.093558576Z INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes

[2025-11-04 01:57:24] DESKTOP-M211L3D:62764:62764 [0] transport/net_ib.cc:526 NCCL WARN Could not find real path of bluerdma0 (/sys/class/infiniband/bluerdma0/device)
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO NET/IB: [0] bluerdma0:uverbs0:1/RoCE provider=None speed=400000 context=0x592d0820dea0 pciPath=(null) ar=0
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO NET/IB : Made virtual device [0] name=bluerdma0 speed=400000 ndevs=1
[2025-11-03T17:57:24.093778735Z DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[2025-11-03T17:57:24.093815101Z DEBUG blue_rdma_driver::verbs::core] before load default
[2025-11-03T17:57:24.093849649Z WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml, using default value
[2025-11-03T17:57:24.093884222Z DEBUG blue_rdma_driver::verbs::core] before open default
[2025-11-03T17:57:24.094041749Z INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[2025-11-03T17:57:24.094073524Z INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes

[2025-11-04 01:57:24] DESKTOP-M211L3D:62764:62764 [0] transport/Made virtual devicenet_ib.cc:526 NCCL WARN Could not find real path of bluerdma1 (/sys/class/infiniband/bluerdma1/device)
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO NET/IB: [1] bluerdma1:uverbs1:1/RoCE provider=None speed=400000 context=0x592d08216950 pciPath=(null) ar=0
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO NET/IB :  [1] name=bluerdma1 speed=400000 ndevs=1
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE [RO]; OOB eth0:100.82.177.43<0>
DESKTOP-M211L3D:62764:62764 [0] NCCL INFO Initialized NET plugin IB
[DESKTOP-M211L3D:62764] *** Process received signal ***
[DESKTOP-M211L3D:62764] Signal: Segmentation fault (11)
[DESKTOP-M211L3D:62764] Signal code: Address not mapped (1)
[DESKTOP-M211L3D:62764] Failing at address: 0x7
[DESKTOP-M211L3D:62764] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x45330)[0x78ad44045330]
[DESKTOP-M211L3D:62764] [ 1] /lib/x86_64-linux-gnu/libnccl.so.2(+0x14e63f)[0x78ad4494e63f]
[DESKTOP-M211L3D:62764] [ 2] /lib/x86_64-linux-gnu/libnccl.so.2(+0x85f89)[0x78ad44885f89]
[DESKTOP-M211L3D:62764] [ 3] /lib/x86_64-linux-gnu/libnccl.so.2(+0x9d7e5)[0x78ad4489d7e5]
[DESKTOP-M211L3D:62764] [ 4] /lib/x86_64-linux-gnu/libnccl.so.2(+0x7f691)[0x78ad4487f691]
[DESKTOP-M211L3D:62764] [ 5] /lib/x86_64-linux-gnu/libnccl.so.2(+0x8e6e7)[0x78ad4488e6e7]
[DESKTOP-M211L3D:62764] [ 6] /lib/x86_64-linux-gnu/libnccl.so.2(pncclCommInitRank+0x1e2)[0x78ad4488ffb2]
[DESKTOP-M211L3D:62764] [ 7] ./nccl_test(+0x98f3)[0x592cd6bff8f3]
[DESKTOP-M211L3D:62764] [ 8] /lib/x86_64-linux-gnu/libc.so.6(+0x2a1ca)[0x78ad4402a1ca]
[DESKTOP-M211L3D:62764] [ 9] /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0x8b)[0x78ad4402a28b]
[DESKTOP-M211L3D:62764] [10] ./nccl_test(+0x9d35)[0x592cd6bffd35]
[DESKTOP-M211L3D:62764] *** End of error message ***
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
^C^Cmake: *** [Makefile:141: run] Interrupt


  sudo apt install libnccl2=2.27.7-1+cuda13.0 libnccl-dev=2.27.7-1+cuda13.0 

