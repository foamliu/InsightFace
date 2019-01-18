# InsightFace

复现 ArcFace [论文](https://arxiv.org/pdf/1801.07698.pdf)

## 数据集

CASIA WebFace 数据集，10,575人物身份，494,414图片。

## 用法

### 数据预处理
提取图片：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```