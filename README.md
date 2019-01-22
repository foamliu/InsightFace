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

#### 图像校准：
原图 | 校准 | 原图 | 校准 |
|---|---|---|---|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/0_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/1_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/2_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/3_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/4_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/5_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/6_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/7_img.jpg)|
|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/8_img.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/InsightFace/raw/master/images/9_img.jpg)|

### 训练
```bash
$ python train.py
```

训练过程可视化：
```bash
$ tensorboard --logdir=runs
```
图 | 损失 | 准确度 |
|---|---|---|
|训练|![image](https://github.com/foamliu/InsightFace/raw/master/images/train_loss.png)|![image](https://github.com/foamliu/InsightFace/raw/master/images/train_acc.png)|
|验证|![image](https://github.com/foamliu/InsightFace/raw/master/images/valid_loss.png)|![image](https://github.com/foamliu/InsightFace/raw/master/images/valid_acc.png)|
