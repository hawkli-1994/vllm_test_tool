# VLLM 测试工具

这是一个用于自动化测试 VLLM 模型的工具。它可以自动启动 Docker 容器，运行 VLLM 服务，并在指定时间后重启服务，用于测试模型的稳定性和性能。该工具特别适用于需要反复测试大模型启动和推理过程的场景。

## 功能特点

- 自动化管理 Docker 容器的启动和停止
- 支持自定义模型路径和配置参数
- 可配置测试迭代次数和冷却时间
- 健康检查确保服务正常运行
- 详细的日志输出和错误追踪
- 自动收集容器日志用于问题诊断

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

基本用法示例：

```bash
python vllm_test.py \
    --model-path /home/user/models/DeepSeek-R1-Distill-Llama-70B \
    --model-name DeepSeek-R1-Distill-Llama-70B \
    --test-iterations 5 \
    --cooldown-time 30
```

### 参数说明

必需参数：
- `--model-path`: 模型文件的完整路径（工具会自动提取父目录作为挂载点）
- `--model-name`: 模型名称

可选参数：
- `--docker-image`: Docker 镜像名称 (默认: rocm/vllm-dev:nightly)
- `--port`: 服务端口 (默认: 8000)
- `--tensor-parallel-size`: 张量并行大小 (默认: 4)
- `--num-scheduler-steps`: 调度器步数 (默认: 8)
- `--max-model-len`: 最大模型长度 (默认: 4096)
- `--max-num-seqs`: 最大序列数 (默认: 512)
- `--test-iterations`: 测试迭代次数 (默认: 5)
- `--cooldown-time`: 冷却时间(秒) (默认: 30)
- `--log-dir`: 日志文件存储目录 (默认: logs)

### 示例

1. 测试 DeepSeek-R1-Distill-Llama-70B 模型（基本用法）：
```bash
python vllm_test.py \
    --model-path /home/user/models/DeepSeek-R1-Distill-Llama-70B \
    --model-name DeepSeek-R1-Distill-Llama-70B
```

2. 自定义配置的完整示例：
```bash
python vllm_test.py \
    --model-path /home/user/models/DeepSeek-R1-Distill-Llama-70B \
    --model-name DeepSeek-R1-Distill-Llama-70B \
    --docker-image rocm/vllm-dev:nightly \
    --port 8000 \
    --tensor-parallel-size 4 \
    --num-scheduler-steps 8 \
    --max-model-len 4096 \
    --max-num-seqs 512 \
    --test-iterations 10 \
    --cooldown-time 60 \
    --log-dir custom_logs
```

## 日志查看

工具会在指定的日志目录（默认为 `logs`）下生成容器日志文件，格式为：
```
logs/container_logs_iter{迭代次数}_{时间戳}.log
```

每个日志文件包含对应迭代中容器的完整输出，包括：
- 服务启动信息
- 错误信息
- 警告信息
- 模型加载状态
- 其他系统信息

## 注意事项

1. 确保有足够的 GPU 内存和系统资源
2. 需要 Docker 环境和适当的权限
3. 确保模型路径正确且可访问
4. 建议在测试前备份重要数据
5. 容器会自动挂载模型目录到 `/app/models`
6. 每次迭代都会保存容器日志，方便问题排查

## 工作原理

工具会：
1. 自动从模型路径提取挂载目录
2. 启动带有正确配置的 Docker 容器
3. 等待服务启动并进行健康检查
4. 收集运行日志
5. 在指定时间后停止容器
6. 重复以上步骤直到完成所有迭代

## 故障排除

如果遇到问题：
1. 检查日志目录中的容器日志文件
2. 确认模型路径和名称是否正确
3. 验证 Docker 权限和 GPU 访问权限
4. 检查端口是否被占用
5. 确保有足够的系统资源 