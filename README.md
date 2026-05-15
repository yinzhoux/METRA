# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

This repository contains the official implementation of **METRA: Scalable Unsupervised RL with Metric-Aware Abstraction**.
The implementation is based on
[Lipschitz-constrained Unsupervised Skill Discovery](https://github.com/seohongpark/LSD).

Visit [our project page](https://seohong.me/projects/metra/) for more results including videos.

## Requirements (原始)
- Python 3.8

## Python 3.10 安装指南

> 原始代码依赖 mujoco-py，仅支持 Python ≤ 3.8。本分支已移植到 **Python 3.10 + native mujoco 2.2.1**，
> 核心算法 (`iod/`) 未做任何修改。

### 系统要求

- Linux（已在 Ubuntu 22.04 验证）
- NVIDIA GPU + CUDA 11.8 或更高（CPU 也可运行，去掉 `+cu118` 后缀改装 CPU 版 torch）
- [Miniconda](https://docs.anaconda.com/miniconda/) 或 Anaconda

### 第一步：创建 conda 环境

```bash
conda create -n metra python=3.10
conda activate metra
```

### 第二步：安装 PyTorch（CUDA 11.8）

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

> CPU 用户：`pip install torch==2.0.1 torchvision==0.15.2`

### 第三步：安装 MuJoCo 和 Gym

```bash
# 使用 native mujoco（不是 mujoco-py）
pip install "mujoco==2.2.1" "gym==0.24.1"
```

> ⚠️ 必须使用 gym==0.24.1。gym 0.25+ 修改了 MujocoEnv 接口，会导致环境初始化失败。

### 第四步：安装其余依赖

```bash
pip install \
    numpy==1.24.3 scipy==1.10.1 matplotlib==3.7.1 \
    akro==0.0.8 \
    wandb==0.15.4 better-exceptions==0.3.3 seaborn==0.12.2 \
    tabulate==0.9.0 tqdm==4.65.0 joblib==1.2.0 pandas==2.0.2 \
    imageio==2.31.0 imageio-ffmpeg==0.4.8 moviepy==1.0.3 \
    cloudpickle==1.3.0 psutil \
    scikit-learn scikit-image==0.25.2 \
    cma==3.3.0 tensorboard==2.10.1 tensorboardX==2.6 \
    absl-py glfw PyOpenGL \
    "setuptools<74"
```

> ⚠️ setuptools 必须 < 74。setuptools 74+ 移除了 `pkg_resources` 模块，而 wandb 0.15.4 依赖它。

### 第五步：安装 garaged 和 iod

```bash
pip install -e garaged --no-deps
pip install -e . --no-deps
```

### 第六步：验证安装

```bash
MUJOCO_GL=egl python -c "
from envs.mujoco.ant_env import AntEnv
from iod.metra import METRA
env = AntEnv()
print('AntEnv OK, obs shape:', env.observation_space.shape)
print('METRA import OK')
"
```

预期输出：
```
AntEnv OK, obs shape: (29,)
METRA import OK
```

---

## 代码兼容性改动说明

为支持 Python 3.10，对以下**非核心**文件做了最小化修改（`iod/` 算法代码未改动）：

| 文件 | 改动原因 |
|------|---------|
| `envs/mujoco/mujoco_utils.py` | 添加 `_SimProxy` + `sim` 属性，将 mujoco-py 风格的 `self.sim.data/model` 代理到 native mujoco 的 `self.data/model` |
| `garagei/envs/consistent_normalized_env.py` | gym 0.24 observation_space 默认 dtype=float64，`flatten` 后需显式 `.astype(np.float32)` 以匹配 PyTorch 模型权重 dtype |
| `garaged/src/garage/tf/__init__.py` | 用 `try/except` 包裹 tensorflow 导入（未安装 TF） |
| `garaged/src/garage/tf/misc/tensor_utils.py` | 同上 |
| `garaged/src/garage/tf/samplers/__init__.py` | 同上 |
| `garaged/src/garage/np/algos/cem.py` | 将 `BatchSampler` 改为懒加载，避免触发 tensorflow 导入链 |
| `garaged/src/garage/np/algos/cma_es.py` | 同上 |

---

## 训练示例

运行前需设置 `MUJOCO_GL=egl`（Linux headless 渲染）：

```bash
# METRA on state-based Ant (2-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2

# LSD on state-based Ant (2-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --dual_reg 0 --spectral_normalization 1 --discrete 0 --dim_option 2

# DADS on state-based Ant (2-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo dads --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# DIAYN on state-based Ant (2-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# METRA on state-based HalfCheetah (16 skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16

# METRA on pixel-based Quadruped (4-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0

# METRA on pixel-based Humanoid (2-D skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0

# METRA on pixel-based Kitchen (24 skills)
MUJOCO_GL=egl python tests/main.py --run_group Debug --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0
```

## 查看训练进度

训练每 100 epochs 向 `exp/<run_group>/<exp_name>/debug.log` 写入一次完整指标，每 1000 epochs 在 `plots/` 目录生成轨迹图和视频。

```bash
# 实时查看日志
tail -f exp/Debug/<exp_name>/debug.log

# TensorBoard
tensorboard --logdir exp/Debug/<exp_name>/tb --port 6006
```
