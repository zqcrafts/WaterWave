# WaterWave: 水下图像/视频增强

这是一个基于 BasicSR 改造的水下增强项目，核心目标是处理水下场景中常见的：

- 颜色偏移（偏蓝/偏绿）
- 低对比度与雾化感
- 细节衰减与纹理模糊
- 序列闪烁（视频帧间不稳定）

当前训练配置以 `WaveField` 模型为主，对应配置文件为 `options/train/0.yml`。

---

## 项目结构（与本工作相关）

- `options/train/0.yml`: 当前主要训练配置（`model_type: WaveField`）
- `basicsr/train.py`: 训练入口
- `basicsr/test.py`: 测试入口
- `scripts/dist_train.sh`: 分布式训练脚本
- `scripts/dist_test.sh`: 分布式测试脚本
- `flow_estimate/`: 光流预处理与相关工具

---

## 环境安装

建议 Python 3.8+，PyTorch 1.10+（按你的 CUDA 版本选择）。

```bash
cd WaterWave
pip install -r requirements.txt
pip install -v -e .
```

---

## 数据准备

`options/train/0.yml` 中默认使用了以下路径：

- `datasets/uw3_mini`（GT/训练数据）
- `dataroot_flow`（光流）
- `dataroot_flow_conf`（光流置信度）

请按你本机实际路径修改 `options/train/0.yml` 里这几项配置后再训练。

---

## 训练

### 单卡训练

```bash
python basicsr/train.py -opt options/train/0.yml
```

### 多卡训练

```bash
bash scripts/dist_train.sh 2 options/train/0.yml
```

---

## 测试 / 验证

### 单卡测试

```bash
python basicsr/test.py -opt options/train/0.yml
```

### 多卡测试

```bash
bash scripts/dist_test.sh 2 options/train/0.yml
```

---

## 结果与日志

训练与测试的输出通常会写入：

- `experiments/<exp_name>/`
- `results/<exp_name>/`

其中 `<exp_name>` 来自配置文件里的 `name`（当前默认为 `wavewater`）。

---

## 注意事项

1. `basicsr/train.py` 中包含一个本地路径追加：
   - `sys.path.append('/home/ubuntu/data/code/waterwave_code')`
   - 如果你本地目录不同，建议改为项目根目录的相对方式，避免跨机器报错。

2. `options/train/0.yml` 当前是一个可跑通模板，实际实验前建议根据你的数据规模调整：
   - `total_iter`
   - `batch_size_per_gpu`
   - `num_frame`
   - 学习率与 scheduler

---

## 致谢

本项目基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 进行二次开发。  
原始框架及相关组件版权与许可请参考仓库内 `LICENSE` 与 `LICENSE.txt`。
