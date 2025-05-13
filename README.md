# VLLM 测试工具

这是一个用于自动化测试 VLLM 模型的工具。它可以自动启动 Docker 容器，运行 VLLM 服务，并在指定时间后重启服务，用于测试模型的稳定性和性能。

## 功能特点

- 自动化管理 Docker 容器的启动和停止
- 支持自定义模型路径和配置参数
- 可配置测试迭代次数和冷却时间
- 健康检查确保服务正常运行
- 详细的日志输出

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

基本用法示例：

```bash
python vllm_test.py \
    --model-path /path/to/model \
    --model-name "model-name" \
    --mount-path /home/user/models \
    --test-iterations 5 \
    --cooldown-time 30
```

### 参数说明

必需参数：
- `--model-path`: 模型文件路径
- `--model-name`: 模型名称
- `--mount-path`: 主机上要挂载到容器的路径

可选参数：
- `--docker-image`: Docker 镜像名称 (默认: rocm/vllm-dev:nightly)
- `--port`: 服务端口 (默认: 8000)
- `--tensor-parallel-size`: 张量并行大小 (默认: 4)
- `--num-scheduler-steps`: 调度器步数 (默认: 8)
- `--max-model-len`: 最大模型长度 (默认: 4096)
- `--max-num-seqs`: 最大序列数 (默认: 512)
- `--test-iterations`: 测试迭代次数 (默认: 5)
- `--cooldown-time`: 冷却时间(秒) (默认: 30)

### 示例

测试 DeepSeek-R1-Distill-Llama-70B 模型：

```bash
python vllm_test.py \
    --model-path /home/user/models \
    --model-name "DeepSeek-R1-Distill-Llama-70B" \
    --mount-path /home/user/models \
    --tensor-parallel-size 4 \
    --num-scheduler-steps 8 \
    --max-model-len 4096 \
    --max-num-seqs 512 \
    --test-iterations 10 \
    --cooldown-time 60
```

## 注意事项

1. 确保有足够的 GPU 内存和系统资源
2. 需要 Docker 环境和适当的权限
3. 确保模型路径正确且可访问
4. 建议在测试前备份重要数据 