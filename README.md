# YOLOv12-DAA: 校园场景大学生摔倒风险预警模型

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 概述

**YOLOv12-DAA** 是一个基于改进 YOLOv12 的轻量化摔倒检测模型，专门面向校园场景（操场、楼道、篮球场、室内等）设计。通过在 YOLOv12 的 3 个 C3k2 特征层中分别集成 **DecoupledCSAttention** 与 **AgentAttention** 双注意力机制，模型能够：

- 解耦通道‑空间特征，强化人体关键部位表征  
- 模拟智能体决策逻辑，动态聚焦高风险动作区域  
- 高效捕捉摔倒动作的瞬时动态时序特征  

在自建校园摔倒数据集 + 公开数据集上，模型取得了 **mAP50 = 99.3%**、**mAP50:95 = 93.3%** 的优异性能，同时保持单帧推理 **0.7 ms**、模型大小仅 **5.4 MB**，满足嵌入式终端实时部署需求。

## 🏗️ 模型架构

![模型整体框架](docs/framework.png)  
*图：YOLOv12-DAA 整体检测流程*

**核心改进点：**

1. **DecoupledCSAttention**  
   - 嵌入 `model.2`（64D）与 `model.20`（256D）层  
   - 独立优化通道与空间注意力，减少背景噪声干扰  

2. **AgentAttention**  
   - 嵌入 `model.4`（128D）层  
   - 通过代理 token 实现线性复杂度全局时序建模，增强快速摔倒动作的特征关联  

3. **残差连接与特征金字塔**  
   - 保留原始特征信息，与双注意力输出融合后送入 FPN 继续检测

## 📊 实验结果

### 主要指标（验证集）

| 模型 | mAP50 (%) | mAP50:95 (%) | 推理时间 (ms) | 模型大小 (MB) |
|------|-----------|--------------|---------------|----------------|
| YOLOv8n | 94.6 | 86.2 | 0.9 | 6.1 |
| YOLOv9n | 95.1 | 87.5 | 1.0 | 6.5 |
| YOLOv10n | 93.8 | 85.0 | 0.8 | 5.9 |
| **YOLOv12-DAA (Ours)** | **99.3** | **93.3** | **0.7** | **5.4** |

### 损失与性能曲线

![训练曲线](docs/training_curves.png)  
*分类损失、边界框回归损失、mAP@50 及 mAP@0.5:0.95 对比*

### 雷达图综合评价

![雷达图](docs/radar.png)  
*YOLOv12-DAA 在精确率、召回率、F1、mAP50、mAP50:95 上均优于其他 YOLO 系列模型*

## 📁 数据集

本项目使用了 **自建校园摔倒数据集** 与两个公开数据集的合并版本：

- **自建数据集**：4213 张标注图像，覆盖跑道、篮球场、室内（瑜伽/武术/日常）等场景  
- **公开数据集**：LE2I-with-Upright-and-Fall、Falldown-Detection-F8XTB  

划分比例：训练集:验证集:测试集 = 7:2:1  

![数据集样本](docs/dataset_samples.png)  
*不同场景下的站立与摔倒样本示例*

## 🚀 快速开始

### 环境要求

- Python 3.11
- PyTorch 2.8.0
- CUDA 12.9 + cuDNN 9.10.2
- 其他依赖：`pip install -r requirements.txt`

### 克隆仓库

```bash
git clone https://github.com/yourusername/YOLOv12-DAA.git
cd YOLOv12-DAA
