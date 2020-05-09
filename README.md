# dipexp
数字图像处理作业（编程部分）

## 小作业

### 图像透视变换

- 读入一幅灰度图像
- 对图像进行透视变换
- 显示结果

[perspective-transform.py](https://github.com/sinofp/dipexp/blob/master/perspective-transform.py)

### 维纳滤波

- 读入一幅灰度图像
- 生成模糊图像：高斯模糊滤波+噪声
- 利用维纳滤波去模糊
- 显示结果

[gaussian&wiener.py](https://github.com/sinofp/dipexp/blob/master/gaussian%26wiener.py)

### 图像分割

- 读入一幅图像
- 进行图像分割
  - 算法不限
  - 建议采用K-means或mean shift
- 显示结果

我选的 K-means ：[image-segmentation-k-mean.py](https://github.com/sinofp/dipexp/blob/master/image-segmentation-k-mean.py)

## 大作业

### 高级图像分割

- 给定一幅图像，进行图像分割
- 要求分割结果要优于作业三

大作业可以调库，但不推荐用神经网络的方法。

我的方法是：
1. 使用felzenszwalb算法过分割图像，得到超级像素
2. 给超级像素建立相似矩阵
3. 用相似矩阵为参数进行谱聚类

[superpixels+spectralcluster.py](https://github.com/sinofp/dipexp/blob/master/superpixels%2Bspectralcluster.py)
