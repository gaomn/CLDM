# Contrastive-Learning-Based Decision Making for Dynamic Time-Linkage Optimization

[![IEEE](https://img.shields.io/badge/IEEE-TSMC-blue.svg)](https://doi.org/10.1109/TSMC.2025.3611797)
[![DOI](https://img.shields.io/badge/DOI-10.1109/TSMC.2025.3611797-blue.svg)](https://doi.org/10.1109/TSMC.2025.3611797)
[![GitHub](https://img.shields.io/badge/GitHub-Open%20Source-green.svg)](https://github.com/gaomn/CLDM)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Language / 语言版本**: [🇨🇳 中文](README.md) | [🇺🇸 English](README_EN.md)

This repository contains the official implementation of the paper **"Contrastive-Learning-Based Decision Making for Dynamic Time-Linkage Optimization"** published in *IEEE Transactions on Systems, Man, and Cybernetics: Systems*.

## Abstract

In dynamic time-linkage optimization, current decisions influence the future state of environments. To make good decisions that have a positive impact on future states, existing methods usually build a model to predict the future rewards of solutions for decision making. However, these prediction models present low accuracy since decision data are not enough to train such a complex model. To address this issue, this article proposes a **contrastive-learning-based decision making (CLDM)** method, which builds a contrastive model to learn the relationship between solutions but not absolute rewards and adopts a quick decision strategy to select solutions.

## 🏗️ Project Structure

```text
cl_pso/
├── components/                 # Core algorithm components
│   ├── CLDM_Model.py          # Contrastive Learning Decision Model
│   ├── PSO.py                 # Particle Swarm Optimization
│   ├── MPB.py                 # Moving Peaks Benchmark
│   ├── Detect.py              # Time-linkage detection
│   ├── RBFN.py                # Radial Basis Function Network
│   ├── Buffer.py              # Data buffer management
│   ├── PPO.py                 # Proximal Policy Optimization
│   └── HybridPER.py           # Hybrid Prioritized Experience Replay
├── all_methods/               # Comparison algorithms
│   ├── cl_pso/                # CL-PSO (Proposed method)
│   ├── drl_pso/               # DRL-PSO (Deep Reinforcement Learning)
│   ├── csvc_pso/              # CSVC-PSO (Support Vector Classifier)
│   ├── pre_pso/               # PRE-PSO (Prediction-based)
│   ├── sql_pso/               # SQL-PSO (Sequential Learning)
│   └── only_pso/              # Standard PSO (Baseline)
├── data_save/                 # Experimental results
├── configs.py                 # Configuration parameters
├── main_loop.py               # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Key Features

- **Contrastive Learning**: Novel approach using relative comparisons instead of absolute reward prediction
- **Time-linkage Detection**: Clustering-based detection to measure time-linkage intensity
- **Quick Decision Strategy**: Efficient solution selection through clustering and ranking
- **Multiple Algorithms**: Implementation of 6 different optimization approaches

## 🔧 Installation

### Prerequisites

- Python 3.9+
- Conda (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/gaomn/CLDM.git
cd CLDM

# Create conda environment
conda create --name cldm python=3.9
conda activate cldm

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

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

## 📊 Usage

### Basic Usage

```bash
# Run CL-PSO with default parameters
python main_loop.py

# Run with custom parameters
python main_loop.py --mode cl-pso --sample_num 5 --using_multiprocessing True
```

### Configuration Parameters

Key parameters in `configs.py`:

- **Environment Parameters**: `max_step`, `x_dim`, `peak_num`, `bt_type`, `Lambda_`
- **PSO Parameters**: `Population_size`, `Iteration_number`, `Inertia_weight`
- **CLDM Parameters**: `cd_threshold`, `k_means_center`, `c`
- **Experimental Parameters**: `sample_num`, `using_multiprocessing`, `b_list`

### Running Comparison Algorithms

```bash
# Navigate to specific algorithm directory
cd all_methods/drl_pso
python main_loop.py --mode drl-pso

cd ../csvc_pso
python main.py --mode csvc

cd ../pre_pso

### Key Parameter Details

All experimental parameters are defined in `configs.py` using `argparse`. You can modify the default values directly in this file or override them via command-line arguments when running.

**Critical Training and Experimental Parameters**:

- `--using_multiprocessing` (bool, default: `False`): Whether to use multiprocessing to run multiple sample experiments in parallel. Setting to `True` can significantly speed up large-scale experiments, but requires attention to CPU and memory resources
- `--bt_change_list` (list, default: `['discrete', 'continuous']`): List of environment change types. Experiments will iterate through each type in this list
- `--Lambda_list` (list, default: `[0.5]`): List of Lambda parameters in MPB environment for controlling change severity
- `--bt_type_list` (list, default: `['linear', 'sin', 'cir']`): List of base environment types
- `--k_means_center_list` (list, default: `[10]`): List of K-Means clustering center numbers for clustering operations in CLDM model
- `--b_list` (list, default: `[10, 50, 100]`): List of time factor b values, affecting the strength of time-linkage characteristics in MPB environment
- `--sample_num` (int, default: `1`): Number of independent runs for each parameter combination. Increasing this value provides more reliable average performance but increases total experiment time

**Parameter Modification Examples**:

```bash
# Modify via command-line arguments
python main_loop.py --Population_size 200 --sample_num 5 --using_multiprocessing True

# Or modify default values directly in configs.py
# parser.add_argument('--Population_size', default=200, type=int)
```

## 📁 Data Storage

Experimental results and generated data are primarily saved in the `data_save/` directory:

**Run Data**:

- Detailed data for each experimental run (such as fitness values and time costs at each step) are saved as CSV files in `data_save/run_data/{timestamp}_{mode}/lambda={lambda_value}/{bt_change_short}_{peak_num}/{bt_type}_b{b_value}/`
- Parameters for each experimental configuration are saved in the `parameters.txt` file in the corresponding directory

**Figures and Other Data**:

- Intermediate figures or specific test data generated in the `run_one` function are saved in `data_save/other_fig/{mode}/step_fig/` and `data_save/other_fig/{mode}/some_data/` directories
python main.py --mode pre
```

## 🧪 Experimental Setup

The experiments are conducted on the Moving Peaks Benchmark (MPB) with various configurations:

- **Environment Types**: Linear, Sinusoidal, Circular
- **Change Types**: Discrete, Continuous
- **Time Factors**: [10, 50, 100]
- **Lambda Values**: [0.5]
- **Peak Numbers**: [1]

Results are automatically saved in the `data_save/` directory with timestamps and parameter configurations.

## 📈 Results

CL-PSO demonstrates good performance in experiments:

- **Better Solution Quality**: Tested on multiple MPB instances
- **Efficient Decision Making**: Quick decision strategy reduces computational overhead
- **Robust Performance**: Consistent results across different dynamic environments

## 🔬 Algorithm Details

### Contrastive Learning Decision Model (CLDM)

1. **Time-linkage Detection**: Uses clustering-based detection to measure environment change intensity
2. **Contrastive Sample Construction**: Creates pairwise comparison samples from limited historical data
3. **RBFN Training**: Trains Radial Basis Function Network for solution comparison
4. **Clustering-based Quick Decision**: Filters and ranks solutions efficiently

### Key Components

- **Detection Module**: Identifies important dimensions and measures change intensity
- **Buffer Management**: Stores and manages historical decision data
- **Contrastive Model**: Learns relative relationships between solutions
- **Quick Decision**: Efficient solution selection strategy

## 📚 Citation

If you use this code in your research, please cite our paper:

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

## 👥 Authors

- **Meng Gao** - Nankai University
- **Xiao-Fang Liu** - Nankai University (Corresponding Author) [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--8137--4201-green.svg)](https://orcid.org/0000-0002-8137-4201)
- **Yongchun Fang** - Nankai University [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--3061--2708-green.svg)](https://orcid.org/0000-0002-3061-2708)
- **Zhi-Hui Zhan** - Nankai University [![ORCID](https://img.shields.io/badge/ORCID-0000--0003--0862--0514-green.svg)](https://orcid.org/0000-0003-0862-0514)
- **Jun Zhang** - Nankai University [![ORCID](https://img.shields.io/badge/ORCID-0000--0001--7835--9871-green.svg)](https://orcid.org/0000-0001-7835-9871)

## 🏛️ Affiliations

- College of Artificial Intelligence, Nankai University, Tianjin, China
- Institute of Intelligence Technology and Robotic Systems, Shenzhen Research Institute of Nankai University, Shenzhen, China
- Hanyang University, ERICA, Ansan, South Korea
- Zhejiang Normal University, Jinhua, China
- Chaoyang University of Technology, Taichung, Taiwan

## 💰 Funding

This work was supported by:

- National Natural Science Foundation of China (NSFC) under Grants 62476141 and 62233011
- National Research Foundation of Korea (NRF) through Korea Government (MSIT) under Grants RS-2025-00555463 and RS-2025-25456394
- Tianjin Top Scientist Studio Project under Grant 24JRRCRC00030
- Tianjin Belt and Road Joint Laboratory under Grant 24PTLYHZ00250

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## ⚠️ Important Notes

- Ensure your computer has sufficient CPU cores and memory to run experiments, especially when `using_multiprocessing` is set to `True` and `sample_num` is large
- The code setting `use_cores = int(num_cores/2)` means it will try to reserve half of the cores from being used by the multiprocessing pool. If your computer has fewer cores, you may need to adjust this value to avoid resource exhaustion
- Experiments may take a long time to run, depending on parameter configuration and number of iterations
- If you encounter compilation errors when installing the minepy package, you can directly use the precompiled wheel file provided in the project directory

## 📞 Contact

For questions about the code or paper, please contact:

- Meng Gao: [gaom@mail.nankai.edu.cn](mailto:gaom@mail.nankai.edu.cn)
- Xiao-Fang Liu: [liuxiaofang@nankai.edu.cn](mailto:liuxiaofang@nankai.edu.cn)

## 🔗 Related Work

- [IEEE Xplore Paper](https://doi.org/10.1109/TSMC.2025.3611797)
- [Nankai University AI Lab](https://ai.nankai.edu.cn/)

---

**Keywords**: Contrastive learning, dynamic time-linkage optimization, evolutionary computation, particle swarm optimization (PSO), prediction

**IEEE Copyright Notice**: © 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses.
