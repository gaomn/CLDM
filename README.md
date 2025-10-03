# 基于对比学习的动态时链优化决策方法

[![IEEE](https://img.shields.io/badge/IEEE-SMAC-blue.svg)](https://doi.org/10.1109/TSMC.2025.3611797)
[![DOI](https://img.shields.io/badge/DOI-10.1109/TSMC.2025.3611797-blue.svg)](https://doi.org/10.1109/TSMC.2025.3611797)
[![GitHub](https://img.shields.io/badge/GitHub-开源项目-green.svg)](https://github.com/gaomn/CLDM)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**语言版本 / Language**: [🇨🇳 中文](README.md) | [🇺🇸 English](README_EN.md)

本仓库包含论文 **"Contrastive-Learning-Based Decision Making for Dynamic Time-Linkage Optimization"** 的官方实现代码，该论文发表于 *IEEE Transactions on Systems, Man, and Cybernetics: Systems*。

## 📖 摘要

在动态时链优化问题中，当前决策会影响环境的未来状态。为了做出对未来状态产生积极影响的良好决策，现有方法通常构建模型来预测解的未来奖励以进行决策。然而，由于决策数据不足以训练如此复杂的模型，这些预测模型的准确性较低。为了解决这个问题，本文提出了一种**基于对比学习的决策方法（CLDM）**，该方法构建对比模型来学习解之间的关系而非绝对奖励，并采用快速决策策略来选择解。

## 🏗️ 项目结构

```text
cl_pso/
├── components/                 # 核心算法组件
│   ├── CLDM_Model.py          # 对比学习决策模型
│   ├── PSO.py                 # 粒子群优化算法
│   ├── MPB.py                 # 动态多峰基准
│   ├── Detect.py              # 时链检测
│   ├── RBFN.py                # 径向基函数网络
│   ├── Buffer.py              # 数据缓冲管理
│   ├── PPO.py                 # 近端策略优化
│   └── HybridPER.py           # 混合优先级经验回放
├── all_methods/               # 对比算法
│   ├── cl_pso/                # CL-PSO（提出方法）
│   ├── drl_pso/               # DRL-PSO（深度强化学习）
│   ├── csvc_pso/              # CSVC-PSO（支持向量分类器）
│   ├── pre_pso/               # PRE-PSO（基于预测）
│   ├── sql_pso/               # SQL-PSO（序列学习）
│   └── only_pso/              # 标准PSO（基线）
├── data_save/                 # 实验结果
├── configs.py                 # 配置参数
├── main_loop.py               # 主执行脚本
├── requirements.txt           # Python依赖
└── README.md                  # 本文件
```

## 🚀 核心特性

- **对比学习**：使用相对比较而非绝对奖励预测的新颖方法
- **时链检测**：基于聚类的检测来测量时链强度
- **快速决策策略**：通过聚类和排序进行高效解选择
- **多种算法**：实现了6种不同的优化方法

### 环境要求

- Python 3.9+
- Conda（推荐）

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/gaomn/CLDM.git
cd CLDM

# 创建conda环境
conda create --name cldm python=3.9
conda activate cldm

# 安装依赖
pip install -r requirements.txt
```

### 依赖包

```text
numpy==1.26.4
tqdm==4.66.2
torch==2.2.1
matplotlib==3.8.3
scikit-learn==1.4.1.post1
statsmodels==0.14.1
scipy==1.12.0
minepy==1.2.6
```

## 📊 使用方法

### 基本使用

```bash
# 使用默认参数运行CL-PSO
python main_loop.py

# 使用自定义参数运行
python main_loop.py --mode cl-pso --sample_num 5 --using_multiprocessing True
```

### 配置参数

`configs.py`中的关键参数：

- **环境参数**：`max_step`、`x_dim`、`peak_num`、`bt_type`、`Lambda_`
- **PSO参数**：`Population_size`、`Iteration_number`、`Inertia_weight`
- **CLDM参数**：`cd_threshold`、`k_means_center`、`c`
- **实验参数**：`sample_num`、`using_multiprocessing`、`b_list`

### 运行对比算法

```bash
# 进入特定算法目录
cd all_methods/drl_pso
python main_loop.py --mode drl-pso

cd ../csvc_pso
python main.py --mode csvc

cd ../pre_pso
python main.py --mode pre
```

## 🧪 实验设置

实验在动态多峰基准（MPB）上进行，配置如下：

- **环境类型**：线性、正弦、圆形
- **变化类型**：离散、连续
- **时间因子**：[10, 50, 100]
- **Lambda值**：[0.5]
- **峰数量**：[1]

结果自动保存在`data_save/`目录中，包含时间戳和参数配置。

## 📈 实验结果

CL-PSO在实验中表现出良好性能：

- **更好的解质量**：在多个MPB实例上进行了测试
- **高效决策**：快速决策策略减少计算开销
- **鲁棒性能**：在不同动态环境中表现一致

## 🔬 算法详情

### 对比学习决策模型（CLDM）

1. **时链检测**：使用基于聚类的检测来测量环境变化强度
2. **对比样本构建**：从有限历史数据创建成对比较样本
3. **RBFN训练**：训练径向基函数网络进行解比较
4. **基于聚类的快速决策**：高效过滤和排序解

### 关键组件

- **检测模块**：识别重要维度并测量变化强度
- **缓冲管理**：存储和管理历史决策数据
- **对比模型**：学习解之间的相对关系
- **快速决策**：高效解选择策略

## 📚 引用格式

如果您在研究中使用了本代码，请引用我们的论文：

```bibtex
@article{liu2025contrastive,
  title={Contrastive-Learning-Based Decision Making for Dynamic Time-Linkage Optimization},
  author={Liu, Xiao-Fang and Gao, Meng and Fang, Yongchun and Zhan, Zhi-Hui and Zhang, Jun},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year={2025},
  volume={PP},
  number={99},
  pages={1--14},
  doi={10.1109/TSMC.2025.3611797},
  publisher={IEEE}
}
```

## 👥 作者信息

- **高猛** - 南开大学
- **刘晓芳** - 南开大学（通讯作者）[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8137--4201-green.svg)](https://orcid.org/0000-0002-8137-4201)
- **方勇纯** - 南开大学 [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3061--2708-green.svg)](https://orcid.org/0000-0002-3061-2708)
- **詹志辉** - 南开大学 [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--0862--0514-green.svg)](https://orcid.org/0000-0003-0862-0514)
- **张军** - 南开大学 [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7835--9871-green.svg)](https://orcid.org/0000-0001-7835-9871)

## 🏛️ 作者单位

- 南开大学人工智能学院，天津，中国
- 南开大学深圳研究院智能技术与机器人系统研究所，深圳，中国
- 韩国汉阳大学ERICA校区，安山，韩国
- 浙江师范大学，金华，中国
- 台湾朝阳科技大学，台中，台湾

## 💰 资助信息

本工作得到以下基金资助：

- 国家自然科学基金（NSFC）资助项目：62476141 和 62233011
- 韩国国家研究基金会（NRF）通过韩国政府（MSIT）资助项目：RS-2025-00555463 和 RS-2025-25456394
- 天津市顶尖科学家工作室项目：24JRRCRC00030
- 天津市"一带一路"联合实验室项目：24PTLYHZ00250

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

## ⚠️ 注意事项

- 确保您的计算机有足够的CPU核心数和内存，特别是在开启`using_multiprocessing`且`sample_num`较大时
- 代码中`use_cores = int(num_cores/2)`会保留一半核心给系统，必要时可根据机器配置调整
- 实验耗时与参数配置和迭代次数有关
- 安装minepy如遇编译问题，可使用项目提供的预编译wheel


如有关于代码或论文的问题，请联系：

- 高猛：[gaom@mail.nankai.edu.cn](mailto:gaom@mail.nankai.edu.cn)
- 刘晓芳：[liuxiaofang@nankai.edu.cn](mailto:liuxiaofang@nankai.edu.cn)

## 🔗 相关链接

- [IEEE Xplore 论文](https://doi.org/10.1109/TSMC.2025.3611797)
- [南开大学人工智能学院](https://ai.nankai.edu.cn/)

---

**关键词**：对比学习，动态时链优化，进化计算，粒子群优化（PSO），预测

**IEEE版权声明**：© 2025 IEEE. 允许个人使用此材料。对于所有其他用途，必须获得IEEE的许可。
