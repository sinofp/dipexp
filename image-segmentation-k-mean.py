import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

K = 3

im = plt.imread("lena512color.tiff")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

ax1.imshow(im)

means = []  # K个点的rgb
clusters = []  # 每个簇的rgb列表，为了求均值
newim = np.zeros(im.shape)  # 新的图片，对应坐标放means
whichcluster = np.zeros((im.shape[0], im.shape[1]), dtype=int)  # 记录对应坐标被归到哪一个类

for i in range(K):
    means.append(
        im[np.random.randint(0, im.shape[0]), np.random.randint(0, im.shape[1])]
    )
    clusters.append([])

iter_times = 0
while True:
    if 10 == iter_times:
        break

    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            rgb = im[x, y]
            distances = []
            for i in range(K):
                distances.append(np.linalg.norm(rgb - means[i]))
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(im[x, y])
            whichcluster[x, y] = cluster_index
            newim[x, y] = means[cluster_index]

    for i in range(K):
        if 0 == len(clusters[i]):
            # 某个类一个元素都没有的话就重现roll一个点的rgb出来
            means[i] = im[
                np.random.randint(0, im.shape[0]), np.random.randint(0, im.shape[1])
            ]
        else:
            means[i] = np.sum(clusters[i], axis=0) / len(clusters[i])

    iter_times += 1

newim = newim.astype(int)
newim[newim > 255] = 255
newim[newim < 0] = 0
ax2.imshow(newim)

colors = "bgrcmykw"
centurycolor = np.zeros(im.shape)
for x in range(whichcluster.shape[0]):
    for y in range(whichcluster.shape[1]):
        centurycolor[x, y] = np.array(mcolors.to_rgb(colors[whichcluster[x, y]]))
ax3.imshow(centurycolor)
