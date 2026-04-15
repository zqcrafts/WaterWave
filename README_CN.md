# WaterWave：水下图像/视频增强

本项目基于 BasicSR 二次开发，面向水下场景增强任务。  
主要解决以下问题：

- 色偏（偏蓝、偏绿）
- 对比度低、雾化严重
- 纹理细节衰减
- 视频帧间闪烁与不稳定

当前主训练配置为 `options/train/0.yml`，模型类型为 `WaveField`。

---

## 目录说明（与当前工作相关）

- `options/train/0.yml`：主训练配置（`model_type: WaveField`）
- `basicsr/train.py`：训练入口
- `basicsr/test.py`：测试入口
- `scripts/dist_train.sh`：分布式训练脚本
- `scripts/dist_test.sh`：分布式测试脚本
- `flow_estimate/`：光流预处理相关代码

---

## 环境安装

建议 Python 3.8+，并按你的 CUDA 版本安装对应的 PyTorch。

```bash
cd WaterWave
pip install -r requirements.txt
pip install -v -e .
```

---

## 数据准备

`options/train/0.yml` 当前使用以下数据项：

- `datasets/uw3_mini`
- `dataroot_flow`
- `dataroot_flow_conf`

请先按你本机路径修改上述配置，再启动训练/测试。

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

## 输出目录

训练与测试结果一般输出到：

- `experiments/<exp_name>/`
- `results/<exp_name>/`

其中 `<exp_name>` 对应配置里的 `name`（当前默认 `wavewater`）。

---

## 注意事项

1. `basicsr/train.py` 中有一行固定路径：
   - `sys.path.append('/home/ubuntu/data/code/waterwave_code')`
   - 若你的本地路径不同，建议改为项目根目录相对路径，避免跨机器报错。

2. `options/train/0.yml` 是当前可运行模板，正式实验建议根据数据规模调整：
   - `total_iter`
   - `batch_size_per_gpu`
   - `num_frame`
   - 学习率与 scheduler

---

## 致谢与许可

本项目基于 [BasicSR](https://github.com/XPixelGroup/BasicSR) 二次开发。  
许可与第三方说明请参考仓库中的 `LICENSE` 与 `LICENSE.txt`。
