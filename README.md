# YOLOv12-DAA：双注意力机制增强的摔倒检测模型

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 简介

**YOLOv12-DAA** 是一个基于 YOLOv12 改进的轻量化摔倒检测模型。通过在 YOLOv12 的不同特征层中分别集成 **DecoupledCSAttention** 和 **AgentAttention** 两种注意力机制，模型能够更精准地捕捉人体摔倒动作的关键特征，同时保持较高的推理效率。

## 模型架构

模型在 YOLOv12 的 3 个 C3k2 特征层中进行了注意力增强：

| 网络层 | 特征维度 | 嵌入的注意力模块 | 作用 |
|--------|----------|------------------|------|
| model.2 | 64D | DecoupledCSAttention | 通道-空间特征解耦，强化人体局部关键部位 |
| model.4 | 128D | AgentAttention | 代理注意力，动态聚焦高风险动作区域 |
| model.20 | 256D | DecoupledCSAttention | 深层特征解耦，提升全局语义表征 |

### 注意力模块说明

- **DecoupledCSAttention**：将通道注意力和空间注意力解耦为两个独立分支，分别优化内容语义与空间位置特征，避免传统联合注意力的信息混淆问题。
- **AgentAttention**：通过少量可学习的代理 token 进行全局特征交互，将注意力计算复杂度降至线性，高效建模摔倒动作的时序动态特征。

## 性能特点

- **高精度**：在包含站立与摔倒两类行为的测试集上，mAP50 达到 99.3%，mAP50:95 达到 93.3%
- **轻量化**：模型大小仅 5.4 MB，适合边缘设备部署
- **实时性**：单帧推理时间 0.7 ms（NVIDIA GeForce RTX 5060 Laptop GPU）
- **低显存**：GPU 显存占用约 480 MB

## 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/YOLOv12-DAA.git
cd YOLOv12-DAA

# 安装依赖
pip install -r requirements.txt
