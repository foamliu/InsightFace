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
#|损失|准确度|
|---|---|---|
|训练|![image](https://github.com/foamliu/InsightFace/raw/master/images/train_loss.png)|![image](https://github.com/foamliu/InsightFace/raw/master/images/train_acc.png)|
|验证|![image](https://github.com/foamliu/InsightFace/raw/master/images/valid_loss.png)|![image](https://github.com/foamliu/InsightFace/raw/master/images/valid_acc.png)|

### 比较
#|图片大小|网络|损失函数|Loss|批量大小|优化器|权重衰减|s|m|预训练|
|---|---|---|---|---|---|---|---|---|---|---|
|1|112x112|ResNet-50|CrossEntropy|4.6201|512|SGD|5e-4|30|0.5|否|

## 性能评估

### LFW
使用 Labeled Faces in the Wild (LFW) 数据集做性能评估:

- 13233 人脸图片
- 5749 人物身份
- 1680 人有两张以上照片
### 准备数据
下载 LFW database 放在 data 目录下:
```bash
$ wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
$ wget http://vis-www.cs.umass.edu/lfw/pairs.txt
$ wget http://vis-www.cs.umass.edu/lfw/people.txt
```