#!/bin/bash

# 设置环境变量
export LD_LIBRARY_PATH=../blue-rdma-driver/dtld-ibverbs/target/debug:../blue-rdma-driver/dtld-ibverbs/rdma-core-55.0/build/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export RUST_LOG=info
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

echo "=== Starting both processes with full debug ==="
echo ""

# 启动服务器（后台）
echo "Starting server (rank 0)..."
./two_process_test 0 > /tmp/server.log 2>&1 &
SERVER_PID=$!

# 等待服务器启动
sleep 2

# 启动客户端
echo "Starting client (rank 1)..."
./two_process_test 1 > /tmp/client.log 2>&1 &
CLIENT_PID=$!

# 等待完成
echo "Waiting for processes to complete..."
wait $SERVER_PID
SERVER_EXIT=$?
wait $CLIENT_PID
CLIENT_EXIT=$?

echo ""
echo "=== Results ==="
echo "Server exit code: $SERVER_EXIT"
echo "Client exit code: $CLIENT_EXIT"
echo ""

echo "=== Server Log (last 50 lines) ==="
tail -50 /tmp/server.log
echo ""

echo "=== Client Log (last 50 lines) ==="
tail -50 /tmp/client.log
echo ""

echo "=== Server WARNINGS ==="
grep "WARN" /tmp/server.log
echo ""

echo "=== Client WARNINGS ==="
grep "WARN" /tmp/client.log
