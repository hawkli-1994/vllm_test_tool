# VLLM 自动化测试工具

本工具旨在简化和自动化 VLLM (Very Large Language Model) 服务的测试流程。它能够自动管理 Docker 容器的生命周期，包括启动、监控、日志收集和关闭，特别适用于需要对大模型进行反复启动和关闭测试的场景，以验证其稳定性和性能。

##核心功能

- **自动化容器管理**：自动处理 Docker 容器的启动、运行监控和停止。
- **灵活参数配置**：支持自定义模型路径、名称、Docker 镜像及多种 VLLM 运行时参数。
- **迭代测试与冷却**：可设定测试迭代次数和每次迭代间的冷却时间。
- **智能健康检查**：通过检查 `/metrics` 和 `/v1/models` 端点，确保 VLLM 服务完全就绪。
- **全面日志收集**：为每个启动的容器生成独立的日志文件，记录从启动到关闭的完整过程。
- **启动超时设置**：允许为模型加载和服务启动配置充裕的等待时间。

## 安装依赖

首先，请确保您已安装 Python 和 pip。然后，通过以下命令安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用指南

### 基本用法

以下命令演示了如何使用默认配置启动测试：

```bash
python vllm_test.py \
    --model-path /path/to/your/model/YourModelName \
    --model-name YourModelName
```

### 参数详解

**必选参数：**

- `--model-path` (str): 指向模型文件的完整路径。脚本会自动提取此路径的父目录作为 Docker 卷的挂载源，并将模型文件挂载到容器内的 `/app/models` 目录下。
  例如：若提供 `/home/user/models/MyLlamaModel`，则 `/home/user/models` 会被挂载。
- `--model-name` (str): 模型的名称，应与 `--model-path` 中指定的模型文件名或目录名一致。
  例如：`MyLlamaModel`。

**可选参数：**

- `--docker-image` (str):  用于运行 VLLM 服务的 Docker 镜像。 (默认: `rocm/vllm-dev:nightly`)
- `--port` (int): VLLM 服务在主机上暴露的端口。 (默认: `8000`)
- `--tensor-parallel-size` (int): VLLM 的张量并行大小。 (默认: `4`)
- `--num-scheduler-steps` (int): VLLM 调度器步数。 (默认: `8`)
- `--max-model-len` (int): 模型支持的最大序列长度。 (默认: `4096`)
- `--max-num-seqs` (int): VLLM 允许的最大并发序列数。 (默认: `512`)
- `--test-iterations` (int): 测试循环的总次数。 (默认: `5`)
- `--cooldown-time` (int): 每次测试迭代结束后，下次启动前的冷却时间（秒）。 (默认: `30`)
- `--log-dir` (str): 存储容器日志文件的目录。 (默认: `logs`)
- `--startup-timeout` (int): 服务启动的最大等待时间（秒）。对于大型模型，建议设置较长的时间。 (默认: `1800`，即 30 分钟)

### 进阶示例

quick start
```
python vllm_test.py \
    --model-path /home/rx/models/DeepSeek-R1-Distill-Llama-70B \
    --model-name DeepSeek-R1-Distill-Llama-70B \
    --test-iterations 5 \
    --cooldown-time 30
```

以下是一个使用自定义参数进行测试的示例：

```bash
python vllm_test.py \
    --model-path /data/models/DeepSeek-R1-Distill-Llama-70B \
    --model-name DeepSeek-R1-Distill-Llama-70B \
    --docker-image custom-vllm-image:latest \
    --port 8001 \
    --tensor-parallel-size 8 \
    --test-iterations 10 \
    --cooldown-time 60 \
    --startup-timeout 3600 \
    --log-dir /var/logs/vllm_tests
```

## 日志管理

- 工具为每次启动的 Docker 容器创建一个独立的日志文件。
- 日志文件命名格式为：`{容器ID前12位}-{启动时间戳}.log` (例如: `a1b2c3d4e5f6-20231026-153000.log`)。
- 所有日志文件默认保存在工作目录下的 `logs` 文件夹中（可通过 `--log-dir` 参数修改）。
- 日志文件会持续记录容器的输出，包括 VLLM 服务的启动信息、运行状态、潜在错误和警告，直至容器停止。

## 注意事项

1. **资源需求**：运行大型模型需要充足的 GPU 显存和系统内存。
2. **Docker 环境**：确保 Docker 已正确安装并运行，且当前用户拥有执行 Docker 命令的权限。
3. **路径准确性**：提供的模型路径必须准确无误，且脚本对其具有读取权限。
4. **数据备份**：虽然本工具不直接修改模型文件，但在进行长时间或高负载测试前，建议备份重要数据。
5. **自动挂载**：脚本会自动将模型所在目录挂载到容器内部的 `/app/models` 路径。

## 工作流程简介

1.  **参数解析**：脚本首先解析用户提供的命令行参数。
2.  **Docker 命令构建**：基于参数构建 `docker run` 命令。
3.  **容器启动**：执行 `docker run` 命令，在后台启动 VLLM 服务容器。
4.  **日志文件创建**：为新启动的容器创建专属的日志文件。
5.  **健康检查与等待**：持续监控服务状态，通过请求 `/metrics` 和 `/v1/models` 端点判断服务是否完全就绪。在此期间，会定期将容器的标准输出和错误流更新到对应的日志文件中。
6.  **稳定运行**：服务启动成功后，保持运行一段时间（由 `--cooldown-time` 定义，此处语义上更像是服务稳定运行观察期，之后脚本会停止容器并进入实际的冷却阶段）。
7.  **日志与停止**：收集最后的日志信息，然后停止并移除 Docker 容器。
8.  **冷却与迭代**：若未达到设定的迭代次数，则进入冷却期，之后开始下一次迭代。

## 故障排查

若测试过程中遇到问题，请尝试以下步骤：

1.  **查阅日志**：仔细检查对应容器的日志文件（位于 `--log-dir` 指定的目录中），通常能找到失败的直接原因。
2.  **核对参数**：确认 `--model-path` 和 `--model-name` 是否正确，模型文件是否存在且可访问。
3.  **Docker 状态**：检查 Docker 服务是否正常运行，以及 GPU 是否对 Docker 可用（如使用 NVIDIA GPU，检查 `nvidia-docker` 或相关配置）。
4.  **端口冲突**：确保 VLLM 服务要使用的端口（默认为 `8000`）未被其他应用占用。
5.  **资源监控**：在测试期间监控系统资源（GPU 显存、内存、CPU），确保没有达到瓶颈。 