import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2

kernel_size = 5
# μ_filter 必 = 0
σ_filter = 5

μ_noise = 0
σ_noise = 1

im = plt.imread("lena512.bmp")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
ax1.imshow(im, cmap="gray")

ax1.set_title("Original")
ax2.set_title("Original + Gaussian filter + Gaussian noise")
ax3.set_title("Original + Gaussian filter + Gaussian noise + Wiener filter")

# 生成高斯核，最后要标准化所以前面×的1/(σ*sqrt(2π))省了
gaussian_kernel = np.fromfunction(
    lambda x, y: np.exp(
        -((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2)
        / (2 * σ_filter ** 2)
    ),
    (kernel_size, kernel_size),
)
gaussian_kernel /= gaussian_kernel.sum()

# 卷积 (转成频域乘法做，快) (但边缘问题呢？)
newim = np.real(ifft2(fft2(im) * fft2(gaussian_kernel, s=im.shape)))

# 高斯噪声
noise = np.random.normal(μ_noise, σ_noise, im.shape)
newim += noise
newim[newim > 255] = 255
newim[newim < 0] = 0

ax2.imshow(newim, cmap="gray")

# 维纳滤波
H = fft2(gaussian_kernel, s=im.shape)
Y = fft2(newim)

# http://www.owlnet.rice.edu/~elec539/Projects99/BACH/proj2/wiener.html
# ↓一种比较简单的估计原图功率谱的方法
Ps = Y * np.conjugate(Y) / newim.shape[0] ** 2
Pn = σ_noise ** 2  # 噪声功率谱是噪声的方差

K = Pn / Ps

G = np.conjugate(H) / (H ** 2 + K)
X = G * Y
reim = np.real(ifft2(X))

ax3.imshow(reim, cmap="gray")
