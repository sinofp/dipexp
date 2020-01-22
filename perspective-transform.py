import numpy as np
import matplotlib.pyplot as plt

# https://www.ece.rice.edu/~wakin/images/
im = plt.imread("lena512.bmp")

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

ax1.imshow(im, cmap="gray")
ax2.imshow(np.zeros(im.shape), cmap="gray")

print("Please click 4 old points")
old_points = plt.ginput(4, timeout=0)
print(old_points)
print("Please click 4 new points")
new_points = plt.ginput(4, timeout=0)
print(new_points)

# https://web.archive.org/web/20150222120106/xenia.media.mit.edu/~cwren/interpolator/
A_list = []
b_list = []
for xy, XY in zip(old_points, new_points):
    A_list.append([xy[0], xy[1], 1, 0, 0, 0, -XY[0] * xy[0], -XY[0] * xy[1]])
    A_list.append([0, 0, 0, xy[0], xy[1], 1, -XY[1] * xy[0], -XY[1] * xy[1]])
    b_list.extend(XY)

a, b, c, d, e, f, g, h = np.linalg.solve(A_list, b_list)
H = np.array([[a, b, c], [d, e, f], [g, h, 1]])

im = im.T  # matplotlib的坐标轴是→x↓y，numpy遍历矩阵是先行后列（↓x→y）
newim = np.zeros(im.shape)
for X in range(newim.shape[0]):
    for Y in range(newim.shape[1]):
        xy1 = np.linalg.solve(H, [X, Y, 1])
        # 其实xy1[2]才应该是1，因为是线性的，乘1/xy1[2]得到正确原图坐标
        x, y, _ = (1 / xy1[2] * xy1).astype(int)

        # 防止越界
        if x >= im.shape[0] or x < 0 or y >= im.shape[1] or y < 0:
            newim[X, Y] = 0
        else:
            newim[X, Y] = im[x, y]

ax2.imshow(newim.T, cmap="gray")
