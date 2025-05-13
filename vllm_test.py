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
import requests
from requests.exceptions import RequestException

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
        log_dir: str = "logs",
        startup_timeout: int = 1800  # 30分钟超时
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
        self.startup_timeout = startup_timeout
        self.log_file: Optional[str] = None
        
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

    def _update_container_logs(self):
        """更新容器日志"""
        if not self.container_id or not self.log_file:
            return
        
        try:
            # 获取容器日志
            log_cmd = ["docker", "logs", self.container_id]
            with open(self.log_file, 'w') as f:
                subprocess.run(log_cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error collecting container logs: {e}")

    def _stop_container(self):
        """停止并删除Docker容器"""
        if self.container_id:
            try:
                # 最后一次更新日志
                self._update_container_logs()
                subprocess.run(["docker", "stop", self.container_id], check=True)
                subprocess.run(["docker", "rm", self.container_id], check=True)
                logger.info(f"Container {self.container_id} stopped and removed")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error stopping container: {e}")
            self.container_id = None
            self.log_file = None

    def _check_container_running(self) -> bool:
        """检查容器是否还在运行"""
        if not self.container_id:
            return False
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().lower() == "true"
        except subprocess.CalledProcessError:
            return False

    def _wait_for_service(self, timeout: int) -> bool:
        """等待服务启动，使用metrics接口进行健康检查"""
        start_time = time.time()
        metrics_url = f"http://localhost:{self.port}/metrics"
        models_url = f"http://localhost:{self.port}/v1/models"
        
        while time.time() - start_time < timeout:
            if not self._check_container_running():
                logger.error("Container stopped unexpectedly")
                return False

            try:
                # 首先检查metrics接口
                metrics_response = requests.get(metrics_url, timeout=5)
                if metrics_response.status_code == 200:
                    # 然后检查models接口
                    models_response = requests.get(models_url, timeout=5)
                    if models_response.status_code == 200:
                        logger.info("Service is fully operational")
                        return True
                    
            except RequestException as e:
                logger.debug(f"Service not ready yet: {str(e)}")
            
            # 每30秒更新一次日志
            if int(time.time() - start_time) % 30 == 0:
                self._update_container_logs()
            
            # 每5秒检查一次服务状态
            time.sleep(5)
            logger.info(f"Waiting for service to start... ({int(time.time() - start_time)}s/{timeout}s)")

        logger.error(f"Service failed to start within {timeout} seconds")
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

                # 创建日志文件
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self.log_file = os.path.join(self.log_dir, f"{self.container_id[:12]}-{timestamp}.log")
                logger.info(f"Container started with ID: {self.container_id}")
                logger.info(f"Container logs will be saved to: {self.log_file}")

                # 等待服务启动并检查健康状态
                if not self._wait_for_service(timeout=self.startup_timeout):
                    logger.error("Service failed to start properly")
                    self._stop_container()
                    continue

                logger.info("Service is running successfully")

                # 运行指定时间
                logger.info(f"Running for {self.cooldown_time} seconds...")
                time.sleep(self.cooldown_time)

                # 停止容器（包含最后的日志更新）
                self._stop_container()

                # 冷却时间
                if iteration < self.test_iterations:
                    logger.info(f"Cooling down for {self.cooldown_time} seconds...")
                    time.sleep(self.cooldown_time)

        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
            self._stop_container()
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
    parser.add_argument('--startup-timeout', type=int, default=1800, help='Timeout for service startup in seconds (default: 30 minutes)')

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
        log_dir=args.log_dir,
        startup_timeout=args.startup_timeout
    )

    tester.run_test()

if __name__ == "__main__":
    main() 