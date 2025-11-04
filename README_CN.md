# MetasurfaceVIT: 光学逆向设计的通用框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

作者：[Jiahao Yan](mailto:yjh20xy@gmail.com)  
[Google Scholar](https://scholar.google.com/citations?user=LSAGvLcAAAAJ&hl=en&oi=ao) | [GitHub](https://github.com/JYJiahaoYan)

---

## 📋 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [项目架构](#项目架构)
- [主要成果](#主要成果)
- [快速开始](#快速开始)
  - [环境安装](#环境安装)
  - [数据生成](#数据生成)
  - [模型预训练](#模型预训练)
  - [超表面设计](#超表面设计)
  - [Jones矩阵重建](#jones矩阵重建)
  - [模型微调与参数预测](#模型微调与参数预测)
  - [前向验证与仿真](#前向验证与仿真)
- [代码文档](#代码文档)
- [常见问题](#常见问题)
- [引用](#引用)
- [许可证](#许可证)

---

## 项目简介

**MetasurfaceVIT** 是一个基于 Vision Transformer 的超表面逆向设计通用框架，专门用于涉及各种振幅和相位工程的超表面设计。该项目通过自监督学习和迁移学习的方式，实现了从目标光学性能到超表面结构参数的端到端预测。

### 主要应用场景

- 🎨 **全息与彩色打印复用**：实现多功能超表面设计
- 🔍 **宽带消色差金属透镜**：跨波长范围的高性能透镜设计
- 🌈 **波长依赖的Jones矩阵工程**：精确控制光的偏振和相位

### 技术亮点

- 采用 **Vision Transformer** 架构处理波长依赖的Jones矩阵数据
- 使用 **SimMIM (Masked Image Modeling)** 进行大规模自监督预训练
- 支持多种 **掩码策略** 以学习不同的光学特性
- 端到端的 **逆向设计流程**：从光学性能到结构参数

---

## 核心特性

### 🔬 完整的工作流程

本项目包含五个主要部分：

1. **数据生成与计算**
   - 基于FDTD仿真生成超表面单元的电磁响应
   - 通过Jones矩阵计算得到波长依赖的光学特性
   - 生成大规模训练数据集（支持~20M数据量）

2. **掩码预训练**
   - 使用波长依赖的Jones矩阵进行自监督预训练
   - 五种掩码策略适应不同的学习目标
   - 支持单GPU和分布式训练

3. **应用导向的超表面设计**
   - 四种设计类型满足不同应用需求
   - 可视化设计结果
   - Jones矩阵重建验证

4. **模型微调与参数预测**
   - 使用预训练模型进行迁移学习
   - 预测超表面的结构参数
   - 支持层级学习率衰减优化

5. **前向验证与光学仿真**
   - 基于前向网络验证预测参数
   - 光学仿真确认设计性能
   - 支持MLP和CNN两种验证网络

### 🎯 模型架构

- **编码器**：Vision Transformer (ViT)
  - 12层Transformer块
  - 512维嵌入向量
  - 12个注意力头
  - 支持绝对位置编码和相对位置偏置
  
- **预训练方法**：SimMIM
  - 掩码图像建模（Masked Image Modeling）
  - 简单的1x1卷积解码器
  - L1重建损失
  
- **微调策略**
  - 冻结部分层或全局微调
  - 层级学习率衰减
  - 结构参数回归头

---

## 项目架构

```
MetasurfaceVIT/
│
├── config.py                      # 配置管理（所有超参数）
├── logger.py                      # 日志系统
├── lr_scheduler.py                # 学习率调度器
├── optimizer.py                   # 优化器构建
├── utils.py                       # 工具函数
│
├── main_pretrain.py               # 预训练主程序
├── main_finetune.py               # 微调主程序
├── main_metalens.py               # 金属透镜设计主程序
│
├── model/                         # 模型定义
│   ├── vision_transformer.py     # Vision Transformer实现
│   ├── simmim.py                 # SimMIM预训练模型
│   └── __init__.py               # 模型构建接口
│
├── data/                          # 数据加载
│   ├── data_simmim.py            # 预训练数据加载器
│   ├── data_finetune.py          # 微调数据加载器
│   ├── data_recon.py             # 重建数据加载器
│   └── __init__.py               # 数据加载接口
│
├── preprocess/                    # 数据预处理
│   ├── data_generation.py        # 数据生成脚本
│   ├── FDTD_Simulation/          # FDTD仿真
│   │   ├── unit_cell.py          # 单元仿真
│   │   ├── prebuilt.fsp          # 预构建仿真文件
│   │   └── unit_script.lsf       # Lumerical脚本
│   └── Jones_matrix_calculation/ # Jones矩阵计算
│       ├── jones_matrix.py       # Jones矩阵运算
│       ├── jones_vector.py       # Jones矢量
│       ├── double_cell.py        # 双单元计算
│       └── visualization.py      # 可视化工具
│
├── evaluation/                    # 评估与验证
│   ├── metasurface_design/       # 超表面设计
│   │   ├── main.py               # 设计主程序
│   │   ├── JM_generator.py       # Jones矩阵生成器
│   │   ├── image_generator.py    # 图像生成器
│   │   └── utils.py              # 工具函数
│   └── metasurface_verification/ # 前向验证
│       ├── main.py               # 验证主程序
│       ├── predictor.py          # 前向预测器
│       ├── matcher.py            # 参数匹配器
│       └── visualization.py      # 可视化
│
├── metalens_output/               # 金属透镜输出
│   ├── lens_construct.lsf        # 透镜构建脚本
│   └── lens_simulate.lsf         # 透镜仿真脚本
│
├── figures/                       # 图片资源
│   ├── presentation/             # 展示图片
│   ├── color/                    # 彩色图片
│   └── grey/                     # 灰度图片
│
├── 代码注释说明.md                 # 中文代码注释文档
└── README_CN.md                   # 中文README（本文件）
```

---

## 主要成果

### 1. 波长依赖的Jones矩阵设计与掩码策略

![设计流程](figures/presentation/fig1.png)

*展示了波长依赖的Jones矩阵表示和五种不同的掩码策略*

### 2. 预训练、设计与重建工作流程

![预训练流程](figures/presentation/fig2.png)

*从数据生成到模型预训练，再到Jones矩阵重建的完整流程*

### 3. 微调、预测与评估工作流程

![微调流程](figures/presentation/fig3.png)

*模型微调和结构参数预测的详细步骤*

### 4. 应用案例1：全息与打印复用

![应用案例1](figures/presentation/fig4.png)

*实现了全息图和彩色打印的复用超表面设计*

### 5. 应用案例2：宽带消色差金属透镜

![应用案例2](figures/presentation/fig5.png)

*跨可见光波段的消色差金属透镜设计*

---

## 快速开始

### 环境安装

#### 1. 系统要求

- Python 3.8+
- CUDA 11.6+ with cuDNN 8
- GPU with 16GB+ VRAM (推荐用于大规模训练)

#### 2. 创建虚拟环境

```bash
# 创建conda环境
conda create -n MetasurfaceVIT python=3.8
conda activate MetasurfaceVIT
```

#### 3. 安装依赖

```bash
# 安装PyTorch (根据你的CUDA版本选择)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
conda install matplotlib pillow numpy scipy
pip install timm termcolor yacs

# 可选：安装Nvidia apex (用于混合精度训练)
# 如果不安装apex，代码会自动使用PyTorch的amp
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

#### 4. 验证安装

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

---

### 数据生成

#### 步骤1：生成预训练数据集

**小规模数据（用于快速测试）：**
```bash
python preprocess/data_generation.py \
    --min_size 40 \
    --max_size 200 \
    --step 20 \
    --points 10 \
    --visualize true
```

**大规模数据（~20M样本，推荐用于正式训练）：**
```bash
python preprocess/data_generation.py
```

参数说明：
- `--min_size`: 结构参数的最小值
- `--max_size`: 结构参数的最大值
- `--step`: 采样步长
- `--points`: 波长采样点数
- `--visualize`: 是否可视化生成的数据

#### 步骤2：生成微调数据集

**小规模微调数据：**
```bash
python preprocess/data_generation.py \
    --min_size 40 \
    --max_size 200 \
    --step 20 \
    --points 10 \
    --visualize true \
    --finetune \
    --finetune_factor 1
```

---

### 模型预训练

#### 单GPU训练（小数据集）

```bash
# 使用Nvidia apex
python main_pretrain.py \
    --epoch 10 \
    --mask_type 0 \
    --data_size 1 \
    --data_start 2

# 或使用PyTorch amp
python main_pretrain.py \
    --epoch 10 \
    --mask_type 0 \
    --data_size 1 \
    --data_start 2 \
    --amp_type pytorch
```

#### 分布式训练（大数据集）

```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    main_pretrain.py \
    --epoch 300 \
    --mask_type 0 \
    --data_size 3 \
    --data_start 1 \
    --batch_size 128
```

#### 掩码类型说明

- `mask_type=0`: 随机选择类型1-5
- `mask_type=1`: 掩码n-1个波长，仅保留一个波长的完整Jones矩阵
- `mask_type=2`: 保留所有振幅，仅保留一个波长的相位
- `mask_type=3`: 类似类型1，但仅保留11极化分量
- `mask_type=4`: 类似类型2，但仅保留11极化分量
- `mask_type=5`: 掩码所有12和22分量，保留所有11分量

---

### 超表面设计

使用预训练模型进行不同类型的超表面设计：

```bash
# 设计类型1
python evaluation/metasurface_design/main.py \
    --pretrain_path preprocess/training_data_2 \
    --design_type 1 \
    --visualize

# 设计类型2
python evaluation/metasurface_design/main.py \
    --pretrain_path preprocess/training_data_2 \
    --design_type 2 \
    --visualize

# 设计类型3
python evaluation/metasurface_design/main.py \
    --pretrain_path preprocess/training_data_2 \
    --design_type 3 \
    --visualize

# 设计类型4（金属透镜）
python evaluation/metasurface_design/main.py \
    --pretrain_path preprocess/training_data_2 \
    --design_type 4 \
    --visualize \
    --amplitude all
```

#### 设计类型说明

- **类型1**: 全息图设计
- **类型2**: 彩色打印设计
- **类型3**: 复用全息与打印
- **类型4**: 宽带消色差金属透镜

---

### Jones矩阵重建

从设计的Jones矩阵重建超表面结构：

```bash
# 重建类型1-3
python main_pretrain.py --recon --recon_type 1
python main_pretrain.py --recon --recon_type 2
python main_pretrain.py --recon --recon_type 3

# 重建类型4（金属透镜）
# 注意：某些情况下（amplitude='all'）可能不需要重建
python main_pretrain.py --recon --recon_type 4
```

---

### 模型微调与参数预测

#### 步骤1：微调预训练模型

**单GPU微调：**
```bash
python main_finetune.py \
    --epoch 100 \
    --data_folder_name finetune_data_1
```

**分布式微调：**
```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    main_finetune.py \
    --epoch 100 \
    --data_folder_name finetune_data_1 \
    --batch_size 64
```

#### 步骤2：结构参数预测

**对于设计类型1-3：**
```bash
python main_finetune.py \
    --eval \
    --data_folder_name finetune_data_1 \
    --recon_type 1 \
    --treatment 2024-10-14

python main_finetune.py \
    --eval \
    --data_folder_name finetune_data_1 \
    --recon_type 2 \
    --treatment 2024-10-14

python main_finetune.py \
    --eval \
    --data_folder_name finetune_data_1 \
    --recon_type 3 \
    --treatment 2024-10-14
```

**对于设计类型4（迭代过程）：**
```bash
python main_metalens.py \
    --eval \
    --data_folder_name finetune_data_1
```

---

### 前向验证与仿真

#### 使用Predictor网络验证

**MLP网络：**
```bash
python evaluation/metasurface_verification/main.py \
    --verify_type predictor \
    --network MLP \
    --train \
    --design_type 1 \
    --treatment 2024-10-14 \
    --finetune_folder finetune_data_1
```

**CNN网络：**
```bash
python evaluation/metasurface_verification/main.py \
    --verify_type predictor \
    --network CNN \
    --train \
    --design_type 1 \
    --treatment 2024-10-14 \
    --finetune_folder finetune_data_1
```

#### 使用Matcher进行验证

```bash
python evaluation/metasurface_verification/main.py \
    --verify_type matcher \
    --design_type 1 \
    --treatment 2024-10-14 \
    --finetune_folder finetune_data_1
```

#### 金属透镜FDTD仿真

对于设计类型4，请导航到`metalens_output/`文件夹并使用Lumerical FDTD进行仿真。

---

## 代码文档

### 中文注释文档

本项目的核心代码文件都已添加详细的中文注释，包括：

#### 已完成注释的文件

1. **配置与工具** (5个文件)
   - `config.py` - 配置管理
   - `logger.py` - 日志系统
   - `optimizer.py` - 优化器构建
   - `lr_scheduler.py` - 学习率调度
   - `utils.py` - 工具函数

2. **核心模型** (3个文件)
   - `model/vision_transformer.py` - ViT实现
   - `model/simmim.py` - SimMIM预训练
   - `model/__init__.py` - 模型接口

3. **数据处理** (2个文件)
   - `data/data_simmim.py` - 预训练数据
   - `data/__init__.py` - 数据接口

详细的代码注释说明请参阅：[代码注释说明.md](代码注释说明.md)

### 注释特点

- ✅ 文件级别的模块说明
- ✅ 类的详细文档字符串
- ✅ 函数的完整参数和返回值说明
- ✅ 关键代码逻辑的行内注释
- ✅ 数据流动和形状变换的标注

---

## 常见问题

### Q1: 如何选择合适的掩码类型？

**A**: 不同的掩码类型适用于不同的学习目标：
- 使用 `mask_type=0` 可以让模型学习所有类型的特征（推荐用于通用预训练）
- 类型1和3侧重于学习波长间的关系
- 类型2和4侧重于学习振幅-相位的关系
- 类型5专注于特定极化分量的学习

### Q2: 预训练需要多长时间？

**A**: 训练时间取决于：
- **数据规模**: 小数据集（~1M）约1-2小时，大数据集（~20M）约1-2天
- **硬件配置**: 单个V100约12小时（20M数据，300 epochs），4卡并行约3-4小时
- **训练轮数**: 建议至少100-300 epochs以获得良好的预训练效果

### Q3: 内存不足怎么办？

**A**: 可以尝试以下方法：
- 减小 `batch_size`（默认128，可降至64或32）
- 使用梯度累积：设置 `--accumulation_steps 2`
- 启用混合精度训练：`--amp_type pytorch`
- 减小模型尺寸：降低 `embed_dim` 或 `depth`

### Q4: 如何验证预训练效果？

**A**: 可以通过以下方式验证：
1. 观察训练损失是否持续下降
2. 使用 `--recon` 模式检查重建的Jones矩阵质量
3. 在微调阶段观察收敛速度（好的预训练应该加速收敛）
4. 比较使用和不使用预训练的最终性能

### Q5: 数据格式是什么？

**A**: 
- **Jones矩阵**: shape为 `[N, 1, wavelengths, 6]`
  - 6个通道对应：`[|J11|, |J12|, |J22|, ∠J11, ∠J12, ∠J22]`
  - wavelengths 通常为20个采样点
- **结构参数**: shape为 `[N, 6]`
  - 6个参数描述超表面单元的几何结构

### Q6: 如何添加自定义的设计类型？

**A**: 
1. 在 `evaluation/metasurface_design/JM_generator.py` 中添加新的设计逻辑
2. 在 `evaluation/metasurface_design/main.py` 中添加对应的命令行选项
3. 确保生成的Jones矩阵格式与训练数据一致

### Q7: 支持哪些学习率调度策略？

**A**: 项目支持：
- **余弦退火** (cosine): 平滑降低学习率，推荐用于大部分场景
- **线性衰减** (linear): 线性降低学习率
- **步进衰减** (step): 每隔固定步数降低学习率
- **多步衰减** (multistep): 在指定的里程碑降低学习率

配置方法见 `config.py` 中的 `TRAIN.LR_SCHEDULER` 部分。

### Q8: 如何使用自己的FDTD仿真数据？

**A**:
1. 确保仿真输出包含S参数（散射矩阵）
2. 使用 `preprocess/Jones_matrix_calculation/` 中的工具转换为Jones矩阵
3. 将数据保存为与项目相同的格式
4. 更新 `config.py` 中的数据路径和参数

---

## 性能优化建议

### 训练优化

1. **使用分布式训练**: 对于大数据集，使用多GPU可以显著加速
   ```bash
   python -m torch.distributed.launch --nproc_per_node 4 main_pretrain.py
   ```

2. **混合精度训练**: 减少内存占用，加速训练
   ```bash
   python main_pretrain.py --amp_type pytorch
   ```

3. **梯度累积**: 在小batch size下模拟大batch效果
   ```bash
   python main_pretrain.py --batch_size 32 --accumulation_steps 4
   ```

4. **层级学习率**: 微调时对不同层使用不同学习率
   - 自动在 `optimizer.py` 中实现
   - 通过 `TRAIN.LAYER_DECAY` 控制衰减率

### 数据处理优化

1. **增加数据加载线程**: `--num_workers 8`（根据CPU核心数调整）
2. **使用PIN内存**: 已默认启用 `PIN_MEMORY=True`
3. **预加载数据**: 对于小数据集，可以考虑全部加载到内存

---

## 引用

如果你在研究中使用了本项目，请引用：

```bibtex
@article{yan2024metasurfacevit,
  title={MetasurfaceVIT: A Generic Framework for Optical Inverse Design},
  author={Yan, Jiahao},
  journal={arXiv preprint},
  year={2024}
}
```

---

## 更新日志

- **2024-10-25**: 初始版本发布，确保小数据集和基本设置可运行
- **2025-04-18**: 更新README，添加主要结果展示
- **2025-11-04**: 添加详细的中文文档和代码注释

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 联系方式

- 作者邮箱：yjh20xy@gmail.com
- GitHub Issues: [提交问题](https://github.com/JYJiahaoYan/MetasurfaceVIT/issues)

---

## 致谢

感谢所有为本项目做出贡献的研究者和开发者。

特别感谢：
- Vision Transformer (ViT) 团队提供的基础架构
- SimMIM 团队提供的自监督学习方法
- Lumerical FDTD Solutions 提供的仿真工具

---

**注意**: 本项目仍在积极开发中，部分功能可能会有更新。请关注 GitHub 仓库获取最新版本。
