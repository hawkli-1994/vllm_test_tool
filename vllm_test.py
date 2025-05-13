#!/usr/bin/env python3

import argparse
import subprocess
import time
import signal
import sys
import os
from typing import Optional
import psutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VLLMTester:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        docker_image: str = "rocm/vllm-dev:nightly",
        port: int = 8000,
        tensor_parallel_size: int = 4,
        num_scheduler_steps: int = 8,
        max_model_len: int = 4096,
        max_num_seqs: int = 512,
        test_iterations: int = 5,
        cooldown_time: int = 30,
        log_dir: str = "logs"
    ):
        self.model_path = os.path.dirname(model_path)  # 获取模型的父目录
        self.model_name = model_name
        self.docker_image = docker_image
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.num_scheduler_steps = num_scheduler_steps
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.test_iterations = test_iterations
        self.cooldown_time = cooldown_time
        self.container_id: Optional[str] = None
        self.log_dir = log_dir
        
        # 创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def _build_docker_command(self) -> list:
        """构建Docker运行命令"""
        return [
            "docker", "run",
            "--network=host",
            "--group-add=video",
            "--ipc=host",
            "--cap-add=SYS_PTRACE",
            "--security-opt", "seccomp=unconfined",
            "--device", "/dev/kfd",
            "--device", "/dev/dri",
            "-v", f"{self.model_path}:/app/models",  # 挂载模型目录到容器的/app/models
            "-p", f"{self.port}:{self.port}",
            "-d",  # 在后台运行
            self.docker_image,
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", f"/app/models/{self.model_name}",  # 使用容器内的路径
            "--served-model-name", self.model_name,
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--num-scheduler-steps", str(self.num_scheduler_steps),
            "--max-model-len", str(self.max_model_len),
            "--max-num-seqs", str(self.max_num_seqs),
            "--enable-prefix-caching"
        ]

    def _collect_container_logs(self, iteration: int):
        """收集容器日志"""
        if not self.container_id:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"container_logs_iter{iteration}_{timestamp}.log")
        
        try:
            # 获取容器日志
            log_cmd = ["docker", "logs", self.container_id]
            with open(log_file, 'w') as f:
                subprocess.run(log_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
            logger.info(f"Container logs saved to {log_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error collecting container logs: {e}")

    def _stop_container(self):
        """停止并删除Docker容器"""
        if self.container_id:
            try:
                subprocess.run(["docker", "stop", self.container_id], check=True)
                subprocess.run(["docker", "rm", self.container_id], check=True)
                logger.info(f"Container {self.container_id} stopped and removed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error stopping container: {e}")
            self.container_id = None

    def _wait_for_service(self, timeout: int = 30) -> bool:
        """等待服务启动，带超时检查"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                health_check_cmd = f"curl -s http://localhost:{self.port}/v1/models"
                result = subprocess.run(health_check_cmd, shell=True, capture_output=True)
                if result.returncode == 0:
                    logger.info("Service is running successfully")
                    return True
                time.sleep(5)
            except subprocess.CalledProcessError:
                time.sleep(5)
        return False

    def run_test(self):
        """运行测试循环"""
        try:
            for iteration in range(1, self.test_iterations + 1):
                logger.info(f"\n=== Starting test iteration {iteration}/{self.test_iterations} ===")
                
                # 启动Docker容器
                cmd = self._build_docker_command()
                logger.info("Starting Docker container...")
                logger.info(f"Command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.container_id = result.stdout.strip()
                
                if not self.container_id:
                    logger.error("Failed to start Docker container")
                    continue

                logger.info(f"Container started with ID: {self.container_id}")

                # 等待服务启动并检查健康状态
                if not self._wait_for_service(timeout=60):
                    logger.error("Service failed to start within timeout")
                    self._collect_container_logs(iteration)
                else:
                    logger.info("Service health check passed")

                # 运行一段时间
                logger.info(f"Running for {self.cooldown_time} seconds...")
                time.sleep(self.cooldown_time)

                # 收集容器日志
                self._collect_container_logs(iteration)

                # 停止容器
                self._stop_container()

                # 冷却时间
                if iteration < self.test_iterations:
                    logger.info(f"Cooling down for {self.cooldown_time} seconds...")
                    time.sleep(self.cooldown_time)

        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
            self._collect_container_logs(iteration)
            self._stop_container()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self._collect_container_logs(iteration)
            self._stop_container()

def main():
    parser = argparse.ArgumentParser(description='VLLM Model Testing Tool')
    parser.add_argument('--model-path', required=True, help='Path to the model directory')
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--docker-image', default='rocm/vllm-dev:nightly', help='Docker image to use')
    parser.add_argument('--port', type=int, default=8000, help='Port to expose')
    parser.add_argument('--tensor-parallel-size', type=int, default=4)
    parser.add_argument('--num-scheduler-steps', type=int, default=8)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--max-num-seqs', type=int, default=512)
    parser.add_argument('--test-iterations', type=int, default=5, help='Number of test iterations')
    parser.add_argument('--cooldown-time', type=int, default=30, help='Cooldown time between iterations (seconds)')
    parser.add_argument('--log-dir', default='logs', help='Directory to store container logs')

    args = parser.parse_args()

    tester = VLLMTester(
        model_path=args.model_path,
        model_name=args.model_name,
        docker_image=args.docker_image,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        num_scheduler_steps=args.num_scheduler_steps,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        test_iterations=args.test_iterations,
        cooldown_time=args.cooldown_time,
        log_dir=args.log_dir
    )

    tester.run_test()

if __name__ == "__main__":
    main() 