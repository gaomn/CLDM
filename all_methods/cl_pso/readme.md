# CL-PSO 项目

## 简介

本项目实现并测试了基于对比学习的粒子群优化 (CL-PSO) 算法，并在动态多峰基准 (Moving Peaks Benchmark, MPB) 问题上进行了测试。该项目旨在研究和评估对比学习策略在动态优化问题中的有效性。

主要特点：

* 实现了多种基于 PSO 的优化算法。
* 使用动态多峰基准问题进行算法评估。
* 支持多参数配置和实验结果的自动保存。
* 利用多进程并行执行实验以提高效率。

## 主要代码文件

* `main_loop.py`: 项目的主执行脚本。它负责初始化参数、设置实验、运行优化算法并保存结果。
* `configs.py`: 定义和管理所有实验参数，使用 `argparse` 库进行配置。
* `components/`: 该目录包含核心算法组件：
  * `Demo.py`: 包含 `CL_PSO`, `PPO_PSO`, `only_PSO` 等高层算法的实现逻辑。
  * `PSO.py`: 标准粒子群优化算法的实现。
  * `MPB.py`: 动态多峰基准 (Moving Peaks Benchmark) 环境的实现。
  * `CLDM_Model.py`: 对比学习决策模型 (CLDM) 的实现，用于 CL-PSO。
  * `PPO.py`: 近端策略优化 (PPO) 代理的实现，用于 PPO-PSO。
  * `Buffer.py`: 数据缓冲区和文件初始化工具。
  * `Detect.py`: 用于变化检测的组件。
  * `RBFN.py`: 径向基函数网络 (RBFN) 的实现。
  * `HybridPER.py`: 混合优先级经验回放的实现。

## 参数修改

所有实验参数都在 `configs.py` 文件中通过 `argparse` 定义。您可以直接修改此文件中的默认值，或者在运行 `main_loop.py` 时通过命令行参数来覆盖它们。

例如，要修改种群大小，您可以：

1. **直接修改 `configs.py`**:
   ```python
   # configs.py
   # ...
   parser.add_argument('--Population_size', default=100, type=int) # 修改这里的默认值
   # ...
   ```
2. **通过命令行参数**:
   ```powershell
   python main_loop.py --Population_size 200
   ```

主要的参数列表包括：

* MPB 环境参数 (`max_step`, `x_dim`, `peak_num`, `bt_type`, `Lambda_` 等)
* PSO 算法参数 (`Population_size`, `Iteration_number`, `Inertia_weight` 等)
* 模型特定参数 (`cd_threshold`, `mode`, `k_means_center` 等)
* 训练和实验参数 (`rand_seed`, `using_multiprocessing`, `sample_num`, 以及各种参数列表如 `peak_num_list`, `bt_type_list` 等)

### 重点训练和实验参数说明

以下是一些关键的训练和实验参数，它们在 `configs.py` 中定义，并对实验的配置和执行有重要影响：

* `--using_multiprocessing` (布尔型, 默认: `False`): 是否使用多进程并行运行多个样本实验。设置为 `True` 可以显著加快大规模实验的速度，但需要注意 CPU 和内存资源。
* `--bt_change_list` (列表, 默认: `['discrete', 'continuous']`): 环境变化类型列表。实验将遍历此列表中的每种变化类型。可选值：`'discrete'` (离散变化), `'continuous'` (连续变化)。
* `--Lambda_list` (列表, 默认: `[0.5]`): MPB 环境中的 `Lambda_` 参数列表，用于控制变化严重程度。实验将遍历此列表中的每个值。
* `--bt_type_list` (字符串列表, 默认: `['linear', 'sin', 'cir']`): 基础环境类型列表。实验将遍历此列表中的每种基础环境类型。可选值：`'linear'`, `'sin'`, `'cir'`。
* `--k_means_center_list` (列表, 默认: `[10]`): K-Means 聚类中心数量列表。用于 CLDM 模型中的聚类操作。实验将遍历此列表中的每个值。
* `--b_list` (列表, 默认: `[10, 50, 100]`): 时间因子 `b` 的列表，影响 MPB 环境中时链特性的强度。实验将遍历此列表中的每个值。
* `--sample_num` (整型, 默认: `1`): 每个参数组合下独立运行的样本数量。增加此值可以获得更可靠的平均性能，但会增加总的实验时间。

这些参数通常在 `main_loop.py` 中通过嵌套循环进行组合，以生成一系列不同的实验配置。修改这些列表的默认值或通过命令行传递新的列表，可以方便地进行多组对比实验。

## 数据保存

实验结果和生成的数据主要保存在 `data_save/` 目录下：

* **运行数据**:
  * 每个实验运行的详细数据（如每一步的适应度值、时间成本等）以 CSV 文件的形式保存在 `data_save/run_data/{timestamp}_{mode}/lambda={lambda_value}/{bt_change_short}_{peak_num}/{bt_type}_b{b_value}/` 路径下。
  * 每个实验配置的参数会保存在对应目录下的 `parameters.txt` 文件中。
* **图形和其他数据**:
  * 在 `run_one` 函数中生成的中间图形或特定测试数据会保存在 `data_save/other_fig/{mode}/step_fig/` 和 `data_save/other_fig/{mode}/some_data/` 目录下。

## 如何运行代码

### 1. 安装 Conda

如果您尚未安装 Conda，请根据您的操作系统从以下链接下载并安装 Miniconda (推荐) 或 Anaconda：

* Miniconda: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
* Anaconda: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

安装完成后，请确保 `conda` 命令可以在您的终端中使用。

### 2. 创建并激活 Conda 环境

打开终端 (例如 Anaconda Prompt, PowerShell, Git Bash 等)，然后执行以下命令：

```powershell
# 进入项目根目录
cd path\to\your\project\all_in_cldm\codes\all_methods\cl_pso

# 创建一个新的 conda 环境 (例如，命名为 cl_pso_env)，并指定 Python 版本 (例如 3.9)
conda create --name cldm python=3.9

# 激活新创建的环境
conda activate cldm
```

### 3. 安装依赖包

在激活的 `cldm` 环境中，使用项目提供的 `requirements.txt` 文件安装所有必需的第三方库：

```powershell
pip install -r requirements.txt
```

注意：如果安装minepy包时遇到编译错误，可以直接使用项目目录中提供的预编译wheel文件进行安装：

```powershell
pip install minepy-1.2.6-cp39-cp39-win_amd64.whl numpy==1.26.4 tqdm==4.66.2 torch==2.2.1 matplotlib==3.8.3 scikit-learn==1.4.1.post1 statsmodels==0.14.1 scipy==1.12.0
```

### 4. 运行代码

一切准备就绪后，您可以运行主脚本 `main_loop.py` 来启动实验：

```powershell
python main_loop.py
```

您可以根据需要在命令后附加参数以修改默认配置，例如：

```powershell
python main_loop.py --mode cl-pso --sample_num 5 --using_multiprocessing True --peak_num_list 1 5 --b_list 10 50
```

### 注意事项

* 确保您的计算机有足够的 CPU核心数 和 内存 来运行实验，特别是当 `using_multiprocessing` 设置为 `True` 且 `sample_num` 较大时。
* 代码中 `use_cores = int(num_cores/2)` 的设置意味着它会尝试保留一半的核心不被多进程池使用。如果您的计算机核心数较少，可能需要调整此值以避免资源耗尽。
* 实验可能需要较长时间运行，具体取决于参数配置和迭代次数。
