# Dst指数预测 - 基于门控机制与物理约束的混合专家系统

## 项目概述
本项目提出了一种用于地磁暴Dst指数预测的**混合专家系统**。通过引入门控机制动态选择最优预测子网络，并针对**正常**与**异常**地磁活动状态分别设计专家网络，实现了对地磁扰动的精准预测。在异常专家的训练中采用**数据增强**提升模型对极端事件的敏感性，在正常专家的训练中引入**物理信息神经网络**增强预测的物理一致性。实验证明，本系统在性能上显著优于单一Conv1DTimeDistributedNet、LSTM等基线模型。

## 背景与动机
Dst指数是衡量地磁暴强度的关键指标，其准确预测对保障卫星通信、电网稳定等基础设施至关重要。然而，现有模型在应对极端事件时往往表现欠佳。本项目基于 **TriQXNet**[1] 的Conv1DTimeDistributedNet架构，结合门控机制和物理约束，旨在提升模型在复杂地磁活动下的鲁棒性和准确性。

## 系统架构
整个系统由三个核心模块构成：**门控网络**、**正常专家** 和 **异常专家**。

### 1. 门控网络
门控网络负责根据输入的时间序列特征（如太阳风参数、黑子数等）动态决定当前样本应分配给哪个专家。
*   **结构**: 结合了CNN（捕捉局部模式）与BiLSTM（捕捉长期依赖），并通过注意力机制聚焦关键时间步。最终通过Softmax层输出选择正常或异常专家的概率。
*   **决策模式**:
    *   **直接门控**: 完全信任门控网络的决策。
    *   **自适应门控**: 仅当门控置信度高于阈值时才遵循其决策，否则默认使用**正常专家**（保守策略），以降低误判风险。

### 2. 正常专家
专门针对地磁活动平静期或普通扰动进行优化。
*   **基础架构**: 基于**TriQXNet**[1] 的Conv1DTimeDistributedNet结构，包含时间维度的全连接层、多层Conv1D、池化层和全连接层。
*   **核心创新 - 物理信息约束**:
    在训练中引入**物理信息神经网络**损失函数，约束预测结果符合**Burton方程**所描述的物理规律：
    > `dDst/dt = α · V · Bs - Dst/τ`
    > 其中 `V` 为太阳风速度，`Bs` 为南向磁场分量。
    > 通过最小化物理残差与数据驱动MSE损失的加权和，使模型预测更具物理意义。

### 3. 异常专家
专门针对强地磁暴等极端事件进行优化。
*   **基础架构**: 同样基于TriQXNet的Conv1DTimeDistributedNet结构。
*   **核心创新 - 数据增强**:
    针对训练集中异常样本稀少的问题，实施了**针对性数据增强**，包括：
    *   **高斯噪声注入**: 增加数据多样性。
    *   **时间抖动**: 模拟时间对齐误差。
    *   **随机缩放**: 模拟事件强度的变化。

### 4. 完整MoE系统
*   **推理流程**:
    1.  输入序列同时送入门控网络与两个专家。
    2.  门控网络输出决策。
    3.  根据选定的模式（直接/自适应），使用对应专家的预测作为最终输出。

## 训练流程
模型分阶段训练，相关脚本位于项目根目录：
1.  **门控网络训练** (`train_gate.py`): 训练一个分类器，用于区分正常与异常样本。
2.  **正常专家训练** (`train_normal.py`): 使用**MSE损失 + PINN物理损失**进行训练。需确保输入数据中包含`speed`和`bz`的索引以计算物理损失。
3.  **异常专家训练** (`train_abnormal.py`): 对原始异常样本进行**数据增强**后，使用标准MSE损失进行训练。

## 实验结果
通过 `test.py` 对训练好的完整MoE系统进行评估，主要结果如下：

| 模式                         | RMSE      | 相对基线改进 | 门控准确率 |
| :--------------------------- | :-------- | :----------- | :--------- |
| **单模型基线 (Conv1DTimeDistributedNet)** | **10.63** | -            | -          |
| **直接门控**                 | **9.72**  | **8.56%**    | 0.9814     |
| **自适应门控 (最佳阈值0.8)** | **9.65**  | **9.22%**    | 0.9814     |
| **完美门控 (理论上限)**      | **9.28**  | **12.70%**   | 1.0        |

*   **结论**: 混合专家系统显著优于单一模型。**自适应门控**通过保守策略进一步提升了性能，缩小了与实际最优（完美门控）的差距。

## 对照模型
为验证本系统（尤其是门控机制和物理损失）的有效性，我们在 `comparison/` 目录下提供了三个对照模型的实现，并完成了对比实验：
*   `LSTM.py`: 基于**TriQXNet**[1] 论文代码改编的标准LSTM模型。
*   `BiLSTM+BiGRU.py`: 基于**Magnet竞赛冠军方案**[2] 改编的双向LSTM+GRU混合模型。
*   `Conv1DTimeDistributedNet.py`: 不含门控架构和PINN损失的单Conv1DTimeDistributedNet模型（作为基线）。

## 环境与依赖
建议使用Python 3.8+ 和 PyTorch 1.9+。
主要依赖项包括：
```
torch
numpy
pandas
matplotlib
tqdm
```

## 使用指南
1.  **数据准备**: 原始数据可从 **NOAA 国家地理数据中心** 的磁强计页面下载：[`https://www.ngdc.noaa.gov/geomag/data/geomag/magnet/`](https://www.ngdc.noaa.gov/geomag/data/geomag/magnet/)。下载后，请确保 `public/` 和 `private/` 目录包含所需的CSV数据文件（如 `dst_labels.csv`, `solar_wind.csv` 等）。代码中已包含完整的数据预处理流程（包括填补、聚合、标准化等）。
2.  **训练模型**: 按顺序运行训练脚本：
    
    ```bash
    python train_gate.py
    python train_normal.py
    python train_abnormal.py
    ```
    训练好的模型权重将分别保存在 `gate_checkpoints/` 和 `expert_checkpoints/` 目录下。
3.  **测试系统**: 运行主测试脚本，将自动加载已训练的模型并输出上述实验结果。
    ```bash
    python test.py
    ```

## 参考文献
[1] Jahin, M. A., et al. (2024). *TriQXNet: Forecasting Dst Index from Solar Wind Data Using an Interpretable Parallel Classical-Quantum Framework with Uncertainty Quantification*. arXiv:2407.06658. [![arXiv](https://img.shields.io/badge/arXiv-2407.06658-b31b1b.svg)](https://arxiv.org/abs/2407.06658)
[2] Magnet Geomagnetic Field Prediction Challenge - 1st Place Solution. [GitHub Repository](https://github.com/drivendataorg/magnet-geomagnetic-field/tree/main/1st_Place)