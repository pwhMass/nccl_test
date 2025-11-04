# NCCL + Blue RDMA Setup Guide for WSL

## 环境检测结果

✅ **Ubuntu 24.04 LTS** (WSL2)
✅ **NVIDIA GPU** (RTX 4060 Ti)
✅ **CUDA 13.0** 已安装于 `/usr/local/cuda-13.0`
❌ **CUDA** 不在 PATH 中（已修复）
❌ **NCCL** 未安装（提供安装脚本）
❌ **MPI** 未安装（提供安装脚本）

---

## 文件修改总结

### 1. **example.cpp** - RCCL → NCCL 转换
**修改内容：**
- HIP/ROCm API → CUDA API
- `rccl.h` → `nccl.h`
- 添加 NCCL IB 传输配置
- 添加详细的错误检查和日志

**关键代码：**
```cpp
// 强制使用 Blue RDMA 设备
setenv("NCCL_IB_DISABLE", "0", 1);     // 启用 IB
setenv("NCCL_NET", "IB", 1);           // 使用 IB 网络
setenv("NCCL_IB_HCA", "bluerdma", 1);  // 指定设备
```

---

### 2. **Makefile** - 自动检测依赖
**新增功能：**
- 自动检测 CUDA 13.0/12.0 安装路径
- 自动检测 NCCL（系统级或自定义安装）
- 新增 `make info` 显示构建配置

**使用示例：**
```bash
make info     # 查看检测到的路径
make MOCK=1   # 构建
make clean    # 清理
```

---

### 3. **run_test.sh** - 智能环境设置
**新增功能：**
- 自动检测 CUDA 安装位置
- 自动添加 CUDA 到 PATH 和 LD_LIBRARY_PATH
- 友好的错误提示

**自动检测顺序：**
1. `/usr/local/cuda-13.0` ← 优先
2. `/usr/local/cuda`
3. `/usr/local/cuda-12.0`

---

### 4. **install_deps.sh** - 自动安装脚本 ⭐
**功能：**
- 自动安装 OpenMPI
- 通过 **Network Repository** 安装 NCCL
- 防止 CUDA 意外升级
- 验证所有组件

**使用方法：**
```bash
chmod +x install_deps.sh
./install_deps.sh
```

**安装内容：**
- OpenMPI (via apt)
- NCCL 2.x (via NVIDIA repository)
- 自动配置依赖

---

### 5. **check_env.sh** - 环境验证脚本 ⭐
**功能：**
- 检查 CUDA、NCCL、MPI
- 检查 Blue RDMA 驱动
- 检查 GPU 可用性
- 提供详细的修复建议

**使用方法：**
```bash
./check_env.sh
```

**示例输出：**
```
✓ CUDA Home: /usr/local/cuda-13.0
✓ nvcc compiler: /usr/local/cuda-13.0/bin/nvcc
✓ NCCL header: /usr/include/nccl.h
  Version: 2.18.5
✓ MPI: Open MPI v4.1.5
✓ Blue RDMA Rust library: ...
```

---

## NCCL 安装方式对比

### Network Repository（推荐，已在脚本中实现）

**特点：**
- ✅ 一行命令完成
- ✅ 自动管理依赖
- ✅ 可通过 apt 更新
- ⚠️ 可能升级 CUDA（脚本已处理）

**安装流程：**
```bash
# 添加 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# 安装 NCCL
sudo apt install libnccl2 libnccl-dev

# 锁定 CUDA 版本（可选）
sudo apt-mark hold cuda-toolkit-13-0
```

---

### Local Repository（离线场景）

**特点：**
- ✅ 离线可用
- ✅ 版本固定
- ❌ 需手动下载
- ❌ 更新麻烦

**安装流程：**
```bash
# 从 NVIDIA 下载 .deb 文件
# https://developer.nvidia.com/nccl/nccl-download

sudo dpkg -i nccl-repo-<version>.deb
# 按提示安装 GPG 密钥
sudo apt update
sudo apt install libnccl2 libnccl-dev
```

---

## 完整使用流程

### Step 1: 检查环境
```bash
cd /home/peng/projects/rdma_all/nccl_test
./check_env.sh
```

### Step 2: 安装依赖
```bash
./install_deps.sh
```
这会安装：
- OpenMPI
- NCCL 2.x (via network repository)

### Step 3: 验证安装
```bash
./check_env.sh
```
确保所有组件都显示 ✓

### Step 4: 查看构建配置
```bash
make info
```
示例输出：
```
===================================
  Build Configuration
===================================
CUDA_HOME:     /usr/local/cuda-13.0
NCCL_INCLUDE:  /usr/include
NCCL_LIB:      /usr/lib/x86_64-linux-gnu
MPI_HOME:      /usr/lib/x86_64-linux-gnu/openmpi
```

### Step 5: 构建测试程序
```bash
make MOCK=1
```

### Step 6: 运行测试
```bash
./run_test.sh
```

---

## 预期结果

### 成功输出示例
```
[Rank 0/2] Process started (PID: 12345)
[Rank 0] NCCL configured to use IB transport with bluerdma device
[Rank 0] NCCL communicator initialized
[Rank 0] GPU memory allocated (4096 bytes)
[Rank 0] Starting ncclAllReduce...
[Rank 0] ncclAllReduce completed
[Rank 0] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 0] Test finished successfully

[Rank 1/2] Process started (PID: 12346)
[Rank 1] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 1] Test finished successfully
```

### 验证要点
1. **NCCL 识别 Blue RDMA 设备**
   - 日志应显示 `bluerdma` 设备
2. **AllReduce 计算正确**
   - 2 个进程：result = 3.0 (1+2)
   - N 个进程：result = N*(N+1)/2
3. **无错误退出**
   - 所有 rank 显示 "Test PASSED"

---

## 故障排除

### 问题 1: CUDA not found
**症状：** `nvcc: command not found`

**解决：**
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

或永久添加到 `~/.bashrc`：
```bash
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 问题 2: NCCL 安装失败
**症状：** `Unable to locate package libnccl2`

**原因：** Ubuntu 24.04 仓库可能还没有 NCCL 包

**解决：** 使用 Ubuntu 22.04 仓库
```bash
# 编辑 install_deps.sh，将 DISTRO 改为：
DISTRO="ubuntu2204"
```

---

### 问题 3: NCCL 检测不到 Blue RDMA 设备
**症状：** NCCL 日志显示使用 TCP/IP 而非 IB

**检查：**
```bash
export LD_LIBRARY_PATH=../dtld-ibverbs/target/debug:../dtld-ibverbs/rdma-core-55.0/build/lib
../dtld-ibverbs/rdma-core-55.0/build/bin/ibv_devices
```

应该看到 `bluerdma` 设备。

---

### 问题 4: GPU 不可用
**症状：** `cudaSetDevice failed`

**WSL GPU 支持检查：**
```bash
nvidia-smi
```

如果无输出，参考：https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute

---

## 技术细节

### NCCL IB 传输流程
```
应用调用 ncclAllReduce()
    ↓
NCCL 选择传输后端 (设置 NCCL_NET=IB)
    ↓
NCCL IB Plugin 调用 ibv_* API
    ↓
libibverbs.so.1 (rdma-core)
    ↓
libbluerdma-rdmav34.so (C Provider)
    ↓
libbluerdma_rust.so (Rust FFI)
    ↓
Blue RDMA Driver (mock 模式软件实现)
```

### Mock 模式说明
- **无需硬件：** 完全在内存中模拟 RDMA 操作
- **无需仿真器：** 不需要 `achronix-400g` 运行
- **适用场景：** API 测试、NCCL 集成测试

---

## 下一步

1. **验证基本功能** ← 当前阶段
2. **测试不同数据大小**
   ```bash
   ./run_test.sh 2 8192   # 2 进程，8K 数据
   ```
3. **添加更多操作**
   - Broadcast
   - Reduce
   - AllGather
4. **性能测试**
   - 使用 NCCL 官方测试套件
   - 测量带宽和延迟

---

## 参考资料

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [NCCL Installation Guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Blue RDMA Architecture](../CLAUDE.md)
- [WSL GPU Support](https://docs.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `example.cpp` | NCCL 测试主程序 |
| `Makefile` | 自动检测依赖的构建文件 |
| `run_test.sh` | 智能测试运行脚本 |
| `install_deps.sh` | NCCL + MPI 自动安装 ⭐ |
| `check_env.sh` | 环境验证脚本 ⭐ |
| `README.md` | 详细文档 |
| `SETUP_GUIDE.md` | 本文档 |

---

**最后更新：** 2025-11-04
**适用环境：** WSL2 + Ubuntu 24.04 + CUDA 13.0
