# NCCL Blue RDMA 测试快速参考

## 一句话总结

| 测试类型 | 一句话说明 | 运行命令 |
|---------|-----------|---------|
| **单进程** | 验证驱动能否被 NCCL 正确加载和初始化 | `make run_single` |
| **双进程** | 验证 RDMA 通信是否正常工作 | Terminal 1: `make run_two_server`<br>Terminal 2: `make run_two_client` |

---

## 单进程测试（5 分钟）

### 快速运行

```bash
cd /home/peng/projects/rdma_all/nccl_test
make run_single
```

### 成功标志

```
=== TEST COMPLETED SUCCESSFULLY ===

Key findings:
  • CUDA runtime: Working
  • GPU memory operations: Working
  • NCCL initialization: Working
  • NCCL IB transport: Detected
  • Blue RDMA driver: Loaded
```

### 日志中的关键信息

#### ✅ 驱动加载成功
```
Setting op alloc_pd
Setting op create_cq
...
bluerdma device allocated
```

#### ✅ 设备发现成功
```
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB
```

#### ✅ 初始化成功
```
[INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
```

### 测试覆盖范围

| 功能 | 是否测试 |
|------|----------|
| 驱动加载 | ✅ |
| 设备查询 | ✅ |
| PD 分配 | ✅ |
| QP 创建 | ❌ |
| 数据通信 | ❌ |

**覆盖率**: ~50%

---

## 双进程测试（10 分钟）

### Terminal 1: 启动服务器

```bash
cd /home/peng/projects/rdma_all/nccl_test
make run_two_server
```

**等待输出**:
```
[Rank 0] Waiting for client connection on port 12345...
```

### Terminal 2: 启动客户端

```bash
cd /home/peng/projects/rdma_all/nccl_test
make run_two_client
```

### 成功标志

**Terminal 1 (Server)**:
```
[Rank 0] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 0] Test completed
```

**Terminal 2 (Client)**:
```
[Rank 1] ✓ Test PASSED: result[0] = 3.0 (expected 3.0)
[Rank 1] Test completed
```

### 测试覆盖范围

| 功能 | 是否测试 |
|------|----------|
| 所有单进程项 | ✅ |
| QP 创建 | ✅ |
| 内存注册 | ✅ |
| post_send | ✅ |
| post_recv | ✅ |
| poll_cq | ✅ |

**覆盖率**: ~95%

---

## 常见问题

### Q: 单进程测试通过了，就说明驱动没问题吗？

**A**: ❌ **不完全是**

单进程测试只验证了：
- 驱动能被加载
- 设备能被发现
- 基本查询操作正常

**但没有测试**：
- RDMA 数据传输
- 完成队列轮询
- QP 状态机

**必须**运行双进程测试才能确认完整功能。

---

### Q: 双进程测试失败了，如何排查？

**A**: 按顺序检查：

#### 1. 单进程测试是否通过？
```bash
make run_single
```
如果失败 → 驱动加载有问题

#### 2. 查看详细日志
```bash
export RUST_LOG=trace
make run_two_server
```

#### 3. 检查是否有 post_send/recv 调用
搜索日志中的：
```
[INFO  bluerdma_rust::rxe::qp_ops] Posting send
[INFO  bluerdma_rust::rxe::qp_ops] Posting recv
```

如果没有 → NCCL 没有调用 RDMA 操作（可能用了 Socket）

#### 4. 检查 Mock 模式是否正常
```bash
cd ../blue-rdma-driver
RUST_LOG=debug cargo test --no-default-features --features mock
```

---

### Q: 为什么要用两个终端？

**A**: 避免 NCCL 的 GPU 重复检测

```
MPI 模式:
mpirun -np 2 ./test
  ↓
  ├─ Process A (GPU 0)
  └─ Process B (GPU 0)  ← NCCL 检测到重复！

独立进程模式:
Terminal 1: ./test 0
Terminal 2: ./test 1
  ↓
  ├─ Process A (GPU 0, 独立进程空间)
  └─ Process B (GPU 0, 独立进程空间)  ← NCCL 无法跨进程检测
```

---

## 测试结果解读

### 单进程测试日志分析

#### ✅ 驱动正确加载
```
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
```

**说明**: 所有 15 个 IB Verbs 操作的函数指针都成功设置

---

#### ✅ 设备成功打开（两次）

```
[DEBUG blue_rdma_driver::verbs::core] before create hardware ctx
[DEBUG blue_rdma_driver::verbs::core] before load default
[WARN  blue_rdma_driver::config] can't load config from /etc/bluerdma/config.toml
[DEBUG blue_rdma_driver::verbs::core] before open default
[INFO  bluerdma_rust::rxe::ctx_ops] Querying device attributes
[INFO  bluerdma_rust::rxe::ctx_ops] Querying port attributes
```

出现两次 → NCCL 打开了 bluerdma0 和 bluerdma1

---

#### ⚠️ 警告可以忽略

```
NCCL WARN Failed to find CUDA library libcuda.so
```
**原因**: WSL2 环境，libcuda.so 在 Windows 侧
**影响**: 无影响，NCCL 会回退到其他机制

```
NCCL WARN Could not find real path of bluerdma0
```
**原因**: Mock 模式没有真实的 PCIe 设备路径
**影响**: 无影响，NCCL 仍能正常工作

---

#### ✅ NCCL 选择了正确的传输

```
NCCL INFO NET/IB : Using [0]bluerdma0:1/RoCE [1]bluerdma1:1/RoCE
NCCL INFO Using network IB
```

**重点**:
- NCCL 识别了 2 个 Blue RDMA 设备
- 选择 IB 作为网络传输（不是 Socket）
- 端口类型显示为 RoCE（RDMA over Converged Ethernet）

---

#### ✅ 通信拓扑构建

```
NCCL INFO comm 0x5cbc082046f0 rank 0 nranks 1 cudaDev 0 busId 1000
NCCL INFO Channel 00/32 :    0
... (32 个通道)
NCCL INFO Trees [0] -1/-1/-1->0->-1
NCCL INFO Connected all rings
NCCL INFO Connected all trees
NCCL INFO 32 coll channels, 0 nvls channels, 32 p2p channels
```

**说明**:
- NCCL 创建了 32 个通道（即使单 rank）
- Ring 和 Tree 拓扑都指向自己（-1->0->-1）
- 这是 NCCL 的标准初始化流程

---

## 调试技巧

### 技巧 1: 逐步增加日志级别

```bash
# 级别 1: 只看结果
make run_single

# 级别 2: 看 NCCL 日志
NCCL_DEBUG=INFO make run_single

# 级别 3: 看 NCCL 详细日志
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL make run_single

# 级别 4: 看驱动日志
RUST_LOG=debug make run_single

# 级别 5: 看驱动详细日志
RUST_LOG=trace make run_single
```

---

### 技巧 2: 隔离测试

```bash
# 只测试 CUDA
./single_gpu_test 2>&1 | grep "CUDA"

# 只测试驱动加载
./single_gpu_test 2>&1 | grep "bluerdma device allocated"

# 只测试 NCCL 后端选择
./single_gpu_test 2>&1 | grep "Using network"
```

---

### 技巧 3: 比对日志差异

```bash
# 保存基线日志
make run_single > baseline.log 2>&1

# 修改代码后
make run_single > current.log 2>&1

# 比对差异
diff baseline.log current.log
```

---

## 性能基准

### 单进程测试（参考值）

| 指标 | 预期值 |
|------|--------|
| 运行时间 | < 5 秒 |
| 内存使用 | < 100 MB |
| GPU 内存 | < 50 MB |

### 双进程测试（参考值）

| 指标 | 预期值 |
|------|--------|
| 运行时间 | < 10 秒 |
| 吞吐量 (Mock) | > 1 GB/s |
| 延迟 | < 10 μs (本地队列) |

---

## 快速命令索引

```bash
# 编译
make all_tests

# 单进程测试
make run_single

# 双进程测试
make run_two_server  # Terminal 1
make run_two_client  # Terminal 2

# 查看配置
make info

# 清理
make clean

# 帮助
make help
```

---

## 测试检查清单

### 单进程测试
- [ ] 编译成功
- [ ] CUDA 设备检测成功
- [ ] 驱动加载成功（"bluerdma device allocated"）
- [ ] NCCL 选择 IB（"Using network IB"）
- [ ] AllReduce 验证通过

### 双进程测试
- [ ] 服务器启动成功
- [ ] 客户端连接成功
- [ ] NCCL ID 交换成功
- [ ] AllReduce 结果正确（3.0）
- [ ] 两个进程都显示 PASSED

---

## 下一步

✅ **单进程测试通过** → 继续双进程测试
❌ **单进程测试失败** → 查看 README.md 故障排查章节

**完整文档**:
- `README.md` - 安装指南
- `TESTING.md` - 详细测试说明
- `TEST_EXPLANATION.md` - 测试原理深入讲解
- `QUICK_REFERENCE.md` - 本文档（快速参考）
