# 单进程模式测试详解

## 你的问题：单进程模式是如何测试的？

这是一个非常好的问题！让我用最清晰的方式解释：

---

## 核心答案

**单进程测试本质上是"半测试"**：
- ✅ 能验证驱动是否正确**加载和初始化**
- ✅ 能验证 NCCL 是否**识别到 Blue RDMA 设备**
- ❌ **不能**测试真正的网络通信（send/recv/poll）

**为什么这样设计？**
- 因为 NCCL 不允许多进程共享一个 GPU
- 单 GPU 系统无法运行传统的双进程 MPI 测试
- 但我们仍需要一个快速验证工具

---

## 详细工作流程

### 1. NCCL 单 Rank 模式的行为

```cpp
// 告诉 NCCL：总共只有 1 个参与者
ncclCommInitRank(&comm, 1, id, 0);
//                      ↑
//                   world_size = 1
```

**NCCL 内部逻辑**（简化版）:

```c
// NCCL 源码伪代码
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank) {
    // 1. 扫描 IB 设备
    int numDevices;
    ibv_device** devices = ibv_get_device_list(&numDevices);  // ← 调用 Blue RDMA

    // 2. 打开设备
    for (int i = 0; i < numDevices; i++) {
        if (match_device_name(devices[i], "bluerdma")) {
            ibv_context* ctx = ibv_open_device(devices[i]);  // ← 调用 Blue RDMA
        }
    }

    // 3. 查询设备属性
    ibv_query_device_ex(ctx, ...);  // ← 调用 Blue RDMA
    ibv_query_port(ctx, 1, ...);    // ← 调用 Blue RDMA

    // 4. 如果 nranks > 1，创建网络资源
    if (nranks > 1) {
        // 分配 PD
        ibv_alloc_pd(ctx);           // ← 多 rank 才调用

        // 注册内存
        ibv_reg_mr(pd, buf, size);   // ← 多 rank 才调用

        // 创建 QP
        ibv_create_qp(pd, ...);      // ← 多 rank 才调用
    } else {
        // nranks == 1: 跳过网络初始化
        // 所有集合操作都是本地内存拷贝
    }

    return ncclSuccess;
}
```

**关键点**:
- NCCL **一定会**调用 `ibv_get_device_list()` 和 `ibv_open_device()`
- NCCL **可能会**创建 QP、注册内存（版本相关）
- NCCL **不会**使用 `post_send`/`post_recv`（因为单 rank 无需网络）

---

### 2. AllReduce 在单 Rank 时的实现

#### 正常情况（2 Ranks）

```
Rank 0                  Rank 1
  ↓                       ↓
[1.0]                   [2.0]
  ↓                       ↓
  ├───── 网络传输 ────────┤
  ↓                       ↓
[3.0]                   [3.0]
  ↑                       ↑
  └─── AllReduce Sum ────┘
```

**RDMA 调用链**:
```
1. ibv_post_send(send 1.0 to Rank 1)
2. ibv_post_recv(recv from Rank 1)
3. ibv_poll_cq(wait for completion)
4. CPU: 1.0 + 2.0 = 3.0
```

#### 单 Rank 情况

```
Rank 0
  ↓
[1.0, 2.0, 3.0]
  ↓
cudaMemcpy(recvbuff, sendbuff, size)  ← 直接内存拷贝！
  ↓
[1.0, 2.0, 3.0]
```

**NCCL 伪代码**:

```c
ncclResult_t ncclAllReduce(void* sendbuff, void* recvbuff, ...) {
    if (comm->nranks == 1) {
        // 单 rank: 直接拷贝
        if (sendbuff != recvbuff) {
            cudaMemcpyAsync(recvbuff, sendbuff, count * sizeof(T), stream);
        }
        return ncclSuccess;
    }

    // 多 rank: 复杂的环状 AllReduce 算法
    // 调用 ibv_post_send, ibv_post_recv, ibv_poll_cq
    ...
}
```

**结果**:
- ❌ **没有调用** `post_send`
- ❌ **没有调用** `post_recv`
- ❌ **没有调用** `poll_cq`
- ✅ **只有** GPU 内存拷贝

---

### 3. 那为什么还有价值？

尽管没有网络通信，单进程测试仍能验证关键功能：

#### 3.1 驱动加载验证

```
NCCL 启动
  ↓
查找 /sys/class/infiniband/ 目录
  ↓
发现 bluerdma0, bluerdma1
  ↓
读取 /sys/class/infiniband_verbs/uverbs0
  ↓
dlopen("libibverbs.so.1")  ← 系统库
  ↓
dlopen("libbluerdma-rdmav34.so")  ← C Provider
  ↓
dlsym("bluerdma_device_alloc")
  ↓
调用 bluerdma_device_alloc()
  ↓
dlopen("libbluerdma_rust.so")  ← Rust FFI
  ↓
dlsym("bluerdma_init")
dlsym("bluerdma_new")
dlsym("bluerdma_alloc_pd")
... (50+ 函数)
  ↓
调用 bluerdma_init()  ← 日志初始化
```

**从日志可以看到**:

```
Setting op alloc_pd           ← dlsym 成功
Setting op create_cq          ← dlsym 成功
Setting op create_qp          ← dlsym 成功
... (50+ 行)
bluerdma device allocated     ← Provider 加载完成
```

✅ **验证了所有 50+ 函数指针都正确导出**

#### 3.2 设备初始化验证

```
bluerdma_new("uverbs0")
  ↓
BlueRdmaCore::new()
  ↓
load_config("/etc/bluerdma/config.toml")  ← 配置加载
  ↓
open_device(Mock/HW)  ← 硬件连接
  ↓
query_device()  ← 查询能力
  ↓
query_port(1)   ← 查询端口状态
```

**从日志可以看到**:

```
[DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[DEBUG blue_rdma_driver::verbs::core] before load default
[WARN  blue_rdma_driver::config] can't load config, using default
[DEBUG blue_rdma_driver::verbs::core] before open default
[INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
```

✅ **验证了驱动初始化流程正常执行**

#### 3.3 NCCL 后端选择验证

```
NCCL 检测可用传输:
  1. 检查 Socket (TCP/IP)
  2. 检查 Plugin (AWS EFA, Azure IB)
  3. 检查 IB Verbs (Blue RDMA)
  ↓
根据性能选择最优传输
```

**从日志可以看到**:

```
NCCL INFO NET/Plugin : Plugin load (libnccl-net.so) returned 2
NCCL INFO NET/Plugin : No plugin found, using internal implementation
NCCL INFO NCCL_IB_HCA set to bluerdma
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB  ← NCCL 选择了 IB！
```

✅ **验证了 NCCL 正确识别并选择 Blue RDMA 作为传输层**

---

## 对比：单进程 vs 双进程

### 测试覆盖对比表

| 函数/操作 | 单进程 | 双进程 | 说明 |
|----------|--------|--------|------|
| `ibv_get_device_list` | ✅ | ✅ | NCCL 初始化时必调用 |
| `ibv_open_device` | ✅ | ✅ | 打开设备 |
| `ibv_query_device_ex` | ✅ | ✅ | 查询设备能力 |
| `ibv_query_port` | ✅ | ✅ | 查询端口状态 |
| `ibv_alloc_pd` | ⚠️ | ✅ | 可能不调用（单 rank 优化） |
| `ibv_reg_mr` | ⚠️ | ✅ | 可能不调用 |
| `ibv_create_cq` | ⚠️ | ✅ | 可能不调用 |
| `ibv_create_qp` | ⚠️ | ✅ | 可能不调用 |
| `ibv_modify_qp` | ❌ | ✅ | 单 rank 不需要 |
| `ibv_post_send` | ❌ | ✅ | 单 rank 不需要 |
| `ibv_post_recv` | ❌ | ✅ | 单 rank 不需要 |
| `ibv_poll_cq` | ❌ | ✅ | 单 rank 不需要 |

**结论**:
- **单进程测试 ≈ 50% 覆盖率**（初始化路径）
- **双进程测试 = 100% 覆盖率**（完整通信路径）

---

## 实际运行演示

### 单进程测试输出解读

```bash
$ make run_single
```

#### 阶段 1: CUDA 初始化

```
✓ Found 1 CUDA device(s)
✓ Using CUDA device 0
  Device name: NVIDIA GeForce RTX 4060 Ti
```
→ CUDA 正常工作

#### 阶段 2: NCCL 配置

```
NCCL INFO NCCL_IB_HCA set to bluerdma
```
→ 环境变量生效

#### 阶段 3: 驱动加载

```
Setting op alloc_pd
Setting op create_cq
... (50+ 行)
bluerdma device allocated
```
→ 所有函数成功导出

#### 阶段 4: 设备打开

```
[DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
```
→ 驱动初始化成功

#### 阶段 5: 后端选择

```
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB
```
→ NCCL 选择了 IB 传输

#### 阶段 6: AllReduce

```
✓ NCCL AllReduce operation posted
✓ NCCL AllReduce completed
✓ NCCL AllReduce result verification: PASSED
```
→ 本地内存拷贝成功（没有网络传输）

---

## 类比：就像测试汽车

### 单进程测试 = 原地启动引擎

```
✅ 检查钥匙能否启动          (驱动加载)
✅ 检查仪表盘显示           (设备查询)
✅ 检查引擎能否运转          (NCCL 初始化)
✅ 检查油门响应             (CUDA 操作)
❌ 但没有真正上路行驶        (没有网络通信)
```

### 双进程测试 = 实际道路测试

```
✅ 所有单进程测试项
✅ 变速箱换挡              (QP 状态转换)
✅ 加速、刹车              (post_send/recv)
✅ 到达目的地              (poll_cq, 数据验证)
```

---

## 总结回答你的问题

**问：单进程模式是如何测试的？**

**答**：

1. **测试范围**：只测试驱动的**初始化路径**，不测试**数据路径**

2. **测试方法**：
   - 创建只有 1 个 rank 的 NCCL communicator
   - NCCL 会加载驱动、打开设备、查询属性
   - 但 AllReduce 操作退化为本地内存拷贝

3. **能验证什么**：
   - ✅ 驱动能否被正确加载（dlopen/dlsym）
   - ✅ 50+ 函数接口是否正确导出
   - ✅ 设备初始化流程是否正常
   - ✅ NCCL 是否识别 Blue RDMA 设备

4. **不能验证什么**：
   - ❌ RDMA send/recv 是否正确
   - ❌ 完成队列轮询是否正确
   - ❌ QP 状态机转换是否正确
   - ❌ Mock 模式的进程间通信

5. **为什么有价值**：
   - 快速验证驱动基础功能（10 秒）
   - 不需要双终端操作
   - 适合快速排查驱动加载问题

6. **推荐使用场景**：
   - 首次编译后快速验证
   - 修改驱动代码后的冒烟测试
   - 排查驱动加载或初始化问题
   - 然后再运行双进程测试做完整验证

**最终建议**：
- 单进程测试 = 入门验证
- 双进程测试 = 完整验证
- 两者配合使用，覆盖完整测试路径
