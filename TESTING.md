# NCCL Blue RDMA Driver Testing Guide

## 测试策略详解

### 为什么需要不同的测试模式？

#### 问题背景

NCCL（NVIDIA Collective Communications Library）有一个重要的限制：
- **每个 NCCL rank 必须使用不同的 GPU**
- 不能让多个进程（MPI ranks）共享同一个 GPU
- 如果检测到多个 rank 使用同一个 GPU，会报错：`Duplicate GPU detected`

```
错误示例：
NCCL WARN Duplicate GPU detected : rank 0 and rank 1 both on CUDA device 1000
NCCL INFO init.cc:1358 -> 5 (ncclInvalidUsage)
```

这个限制导致在**单 GPU 系统**上无法使用传统的 MPI 多进程测试。

---

## 测试模式对比

### 1. 单进程测试 (Single Process Test)

**文件**: `single_gpu_test.cpp`

**运行**: `make run_single`

#### 工作原理

```cpp
// 初始化 NCCL：只有 1 个 rank
ncclCommInitRank(&comm, 1, id, 0);
//                      ↑     ↑
//                   总数=1  rank=0

// AllReduce 操作：单 rank 时 = 数据复制
ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream);
// 输入: [1.0, 2.0, 3.0]
// 输出: [1.0, 2.0, 3.0] (无其他 rank 参与，直接复制)
```

#### 测试覆盖范围

| 功能模块 | 是否测试 | 具体内容 |
|---------|---------|---------|
| ✅ CUDA 运行时 | **是** | `cudaMalloc`, `cudaMemcpy`, `cudaMemset` |
| ✅ NCCL 初始化 | **是** | `ncclGetUniqueId`, `ncclCommInitRank` |
| ✅ Blue RDMA 发现 | **是** | NCCL 调用 `ibv_get_device_list()` |
| ✅ 设备查询 | **是** | `query_device`, `query_port` |
| ✅ Provider 加载 | **是** | `dlopen("libbluerdma-rdmav34.so")` |
| ⚠️ 队列创建 | **可能** | NCCL 可能创建 QP，但不使用 |
| ❌ 网络通信 | **否** | 没有实际的 send/recv |
| ❌ RDMA post_send | **否** | 单 rank 无需网络传输 |
| ❌ RDMA post_recv | **否** | 单 rank 无需网络传输 |
| ❌ 完成队列轮询 | **否** | 没有 `poll_cq` |
| ❌ 内存注册 | **可能** | 可能注册内存但不进行 RDMA |

#### 日志分析

从日志可以看到驱动加载过程：

```
NCCL INFO NCCL_IB_HCA set to bluerdma
↓
Setting op alloc_pd
Setting op create_cq
Setting op create_qp
... (50+ 操作)
bluerdma device allocated
↓
[DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[DEBUG blue_rdma_driver::verbs::core] before load default
[INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
↓
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB
```

**结论**:
- ✅ 驱动成功加载
- ✅ 设备发现和查询正常
- ✅ NCCL 选择 IB 作为网络后端
- ❌ 但没有真正的网络数据传输

#### 价值

这个测试可以验证：
1. 驱动的**接口正确性**（所有 50+ 函数都能正确导出）
2. 驱动的**初始化流程**（日志系统、配置加载、设备打开）
3. NCCL 的**后端选择逻辑**（确认使用 IB 而不是 Socket）
4. CUDA 和驱动的**兼容性**

---

### 2. 双进程测试 (Two-Process Test)

**文件**: `two_process_test.cpp`

**运行**:
```bash
# Terminal 1
make run_two_server

# Terminal 2
make run_two_client
```

#### 工作原理

##### 架构图

```
┌─────────────────────┐          ┌─────────────────────┐
│   Process 1         │          │   Process 2         │
│   (Rank 0)          │          │   (Rank 1)          │
│                     │          │                     │
│  GPU 0              │          │  GPU 0 (相同)        │
│  ├─ CUDA Context 1  │          │  ├─ CUDA Context 2  │
│  ├─ NCCL Comm 1     │          │  ├─ NCCL Comm 2     │
│  └─ Blue RDMA 0 ────┼──────────┼──┤  Blue RDMA 1     │
│      17.34.51.10    │  Network │  │      17.34.51.11 │
└─────────────────────┘          └─────────────────────┘
         ↑                                    ↑
         └────── TCP Socket (NCCL ID) ───────┘
```

##### 关键设计

1. **不使用 MPI**：避免 GPU 重复检测
   ```cpp
   // 不用 MPI_Init()
   // 改用 TCP socket 交换 NCCL ID
   exchangeNcclId(rank, &id);
   ```

2. **使用不同的 RDMA 设备**：
   ```cpp
   if (rank == 0) {
       setenv("NCCL_IB_HCA", "bluerdma0", 1);
   } else {
       setenv("NCCL_IB_HCA", "bluerdma1", 1);
   }
   ```

3. **独立进程空间**：
   - 手动启动两个进程
   - 每个进程有独立的 CUDA context
   - NCCL 无法检测到 GPU 重复（不同进程）

4. **TCP 握手**：
   ```cpp
   // Rank 0: 生成 NCCL ID 并监听
   ncclGetUniqueId(&id);
   int server_fd = socket(...);
   bind(server_fd, ...);
   listen(server_fd, 1);
   int client_fd = accept(...);
   send(client_fd, &id, sizeof(ncclUniqueId), 0);

   // Rank 1: 连接并接收 NCCL ID
   int sock = socket(...);
   connect(sock, "127.0.0.1:12345", ...);
   recv(sock, &id, sizeof(ncclUniqueId), 0);
   ```

5. **NCCL 初始化**：
   ```cpp
   // 两个进程都初始化为 2-rank 系统
   ncclCommInitRank(&comm, 2, id, rank);
   //                      ↑
   //                   总共 2 个 rank
   ```

6. **数据传输**：
   ```cpp
   // Rank 0: sendbuff = [1.0, 1.0, ...]
   // Rank 1: sendbuff = [2.0, 2.0, ...]

   ncclAllReduce(sendbuff, recvbuff, size, ncclFloat, ncclSum, comm, stream);

   // 结果: recvbuff = [3.0, 3.0, ...] (1.0 + 2.0)
   ```

#### 测试覆盖范围

| 功能模块 | 是否测试 | 具体内容 |
|---------|---------|---------|
| ✅ 所有单进程测试项 | **是** | 见上表 |
| ✅ 网络通信 | **是** | NCCL 通过 IB 传输数据 |
| ✅ RDMA post_send | **是** | Blue RDMA 的 `post_send()` |
| ✅ RDMA post_recv | **是** | Blue RDMA 的 `post_recv()` |
| ✅ 完成队列轮询 | **是** | Blue RDMA 的 `poll_cq()` |
| ✅ 内存注册 | **是** | Blue RDMA 的 `reg_mr()` |
| ✅ QP 状态机 | **是** | `modify_qp()` (INIT→RTR→RTS) |
| ✅ 地址交换 | **是** | NCCL 通过 IB GID/LID 建立连接 |
| ✅ Mock 模式通信 | **是** | 进程间内存队列 |

#### 预期调用链

```
1. ncclCommInitRank()
   ↓
2. NCCL 扫描 IB 设备 → ibv_get_device_list()
   ↓
3. 打开设备 → ibv_open_device() → bluerdma_new()
   ↓
4. 分配 PD → ibv_alloc_pd() → bluerdma_alloc_pd()
   ↓
5. 注册内存 → ibv_reg_mr() → bluerdma_reg_mr()
   ↓
6. 创建 CQ → ibv_create_cq() → bluerdma_create_cq()
   ↓
7. 创建 QP → ibv_create_qp() → bluerdma_create_qp()
   ↓
8. 修改 QP 状态 → ibv_modify_qp() → bluerdma_modify_qp()
   - RESET → INIT
   - INIT → RTR (Ready To Receive)
   - RTR → RTS (Ready To Send)
   ↓
9. ncclAllReduce()
   ↓
10. NCCL 发送数据 → ibv_post_send() → bluerdma_post_send()
    ↓ (Mock 模式: 写入共享内存队列)
    ↓ (HW 模式: 写 CSR 寄存器)
    ↓
11. NCCL 接收数据 → ibv_post_recv() → bluerdma_post_recv()
    ↓
12. NCCL 轮询完成 → ibv_poll_cq() → bluerdma_poll_cq()
    ↓ (Mock 模式: 检查内存队列)
    ↓ (HW 模式: 读 CSR 寄存器)
    ↓
13. cudaStreamSynchronize() → 等待所有操作完成
```

---

## 测试执行步骤

### 步骤 1: 单进程测试（快速验证）

```bash
cd /home/peng/projects/rdma_all/nccl_test

# 编译
make all_tests

# 运行单进程测试
make run_single
```

**预期输出**:

```
=== Single GPU NCCL Test for Blue RDMA Driver ===

✓ NCCL configured to use IB transport with bluerdma device
✓ Found 1 CUDA device(s)
✓ Using CUDA device 0
  Device name: NVIDIA GeForce RTX 4060 Ti
  Compute capability: 8.9
  Total global memory: 15.73 GB

--- Testing CUDA Operations ---
✓ Allocated 4 MB on GPU
✓ Memory set successful
✓ Memory copy successful
✓ Data verification: PASSED

--- Testing NCCL Initialization ---
✓ Generated NCCL unique ID
✓ NCCL communicator initialized (single rank mode)
✓ Allocated buffers for NCCL operations
✓ NCCL AllReduce operation posted
✓ NCCL AllReduce completed
✓ NCCL AllReduce result verification: PASSED

--- Cleanup ---
✓ All resources cleaned up

=== TEST COMPLETED SUCCESSFULLY ===

Key findings:
  • CUDA runtime: Working
  • GPU memory operations: Working
  • NCCL initialization: Working
  • NCCL IB transport: Detected (check logs above)
  • Blue RDMA driver: Loaded (check 'Setting op' messages above)
```

**如果失败**:
- 检查 CUDA 驱动版本（nvidia-smi）
- 检查 LD_LIBRARY_PATH 设置
- 查看 RUST_LOG=debug 输出

---

### 步骤 2: 双进程测试（完整验证）

#### Terminal 1: 启动服务器

```bash
cd /home/peng/projects/rdma_all/nccl_test
make run_two_server
```

**预期输出**:

```
Starting server (rank 0)...
Waiting for client connection...
=== Two-Process NCCL Test for Blue RDMA Driver ===
Process: Rank 0

[Rank 0] Using Blue RDMA device 0
[Rank 0] CUDA initialized on device 0
[Rank 0] Waiting for client connection on port 12345...
```

**卡住是正常的**，等待客户端连接。

#### Terminal 2: 启动客户端

```bash
cd /home/peng/projects/rdma_all/nccl_test
make run_two_client
```

**预期输出（Terminal 2）**:

```
Starting client (rank 1)...
Connecting to server...
=== Two-Process NCCL Test for Blue RDMA Driver ===
Process: Rank 1

[Rank 1] Using Blue RDMA device 1
[Rank 1] CUDA initialized on device 0
[Rank 1] Connecting to server on localhost:12345...
[Rank 1] NCCL ID received from server
[Rank 1] NCCL communicator initialized
[Rank 1] Initialized send buffer with value 2.0
[Rank 1] Starting AllReduce operation...

... (大量 NCCL 和 Blue RDMA 日志) ...

[Rank 1] AllReduce completed
[Rank 1] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 1] Test completed
```

**同时在 Terminal 1 看到**:

```
[Rank 0] NCCL ID sent to client
[Rank 0] NCCL communicator initialized
[Rank 0] Initialized send buffer with value 1.0
[Rank 0] Starting AllReduce operation...

... (大量 NCCL 和 Blue RDMA 日志) ...

[Rank 0] AllReduce completed
[Rank 0] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 0] Test completed
```

---

## 日志分析要点

### 关键成功指标

#### 1. Driver 加载

```
Setting op alloc_pd
Setting op create_cq
Setting op create_qp
... (50+ 行)
bluerdma device allocated
```
✅ **所有操作都设置成功**

#### 2. 设备初始化

```
[DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[DEBUG blue_rdma_driver::verbs::core] before load default
[WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml
[DEBUG blue_rdma_driver::verbs::core] before open default
```
✅ **驱动初始化流程执行**（配置文件缺失是正常的，使用默认值）

#### 3. NCCL 选择 IB

```
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB
```
✅ **NCCL 识别到 Blue RDMA 设备并选择 IB 传输**

#### 4. 通信成功（仅双进程测试）

```
[Rank 0] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 1] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
```
✅ **AllReduce 计算正确：1.0 + 2.0 = 3.0**

---

## 常见问题

### Q1: 单进程测试通过，但双进程测试失败？

**可能原因**:
- Mock 模式的共享内存队列有问题
- RDMA post_send/post_recv 实现错误
- QP 状态机转换失败

**排查方法**:
```bash
# 查看详细日志
export RUST_LOG=trace
make run_two_server
```

### Q2: 双进程测试卡住不动？

**可能原因**:
- 客户端连接失败（端口被占用）
- NCCL 初始化超时
- Blue RDMA 死锁

**排查方法**:
```bash
# 检查端口
netstat -tlnp | grep 12345

# 查看进程状态
ps aux | grep two_process_test

# 强制停止
killall two_process_test
```

### Q3: 为什么不能用 MPI？

**原因**: NCCL 有特殊的 GPU 检测逻辑

```c
// NCCL 源码（简化）
for (int i = 0; i < nranks; i++) {
    for (int j = i+1; j < nranks; j++) {
        if (ranks[i].busId == ranks[j].busId) {
            WARN("Duplicate GPU detected: rank %d and rank %d both on CUDA device %lx",
                 i, j, ranks[i].busId);
            return ncclInvalidUsage;
        }
    }
}
```

MPI 启动的所有进程都在**同一个进程树**下，NCCL 可以扫描所有 rank 的 GPU busId。

我们的双进程测试使用**完全独立的进程**，NCCL 无法跨进程扫描。

---

## 总结

| 测试类型 | 适用场景 | 测试深度 | 运行复杂度 |
|---------|---------|---------|-----------|
| **单进程测试** | 快速验证驱动加载 | ⭐⭐☆☆☆ | ⭐☆☆☆☆ |
| **双进程测试** | 完整网络通信测试 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐☆☆ |
| **MPI 测试** | ❌ 不可用（单 GPU） | - | - |

**推荐测试流程**:
1. 先运行 `make run_single` 确保基础功能正常
2. 再运行双进程测试验证完整通信路径
3. 如果有多个 GPU，可以尝试修改 MPI 测试使用不同 GPU

**最终目标**: 所有测试都显示 `✓ Test PASSED`
