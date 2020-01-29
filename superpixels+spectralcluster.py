import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import skimage.segmentation as seg
from sklearn.cluster import spectral_clustering
from skimage import color
from sys import argv

if (len(argv) < 3):
    print("Need input filename and output filename.")
    exit()

inputfile, outputfile = argv[1:]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.set_title("Original")
ax2.set_title("Original + Felzenszwalb")
ax3.set_title("Original + Felzenszwalb + Spectral Clustering")
ax4.set_title("Original + Felzenszwalb + Spectral Clustering (average color)")

im = plt.imread(inputfile)

ax1.imshow(im)

labels_fz = seg.felzenszwalb(im, 200, sigma=2, min_size=100)

ax2.imshow(seg.mark_boundaries(im, labels_fz))

G = nx.Graph()
G.add_nodes_from(range(len(np.unique(labels_fz))))
labels_fz_x_size, labels_fz_y_size = labels_fz.shape

for i in range(labels_fz_x_size):
    for j in range(labels_fz_y_size):

        for dx, dy in zip([1, 0, -1, 0], [0, -1, 0, 1]):
            x = i + dx
            y = j + dy

            x = min(max(0, x), labels_fz_x_size - 1)
            y = min(max(0, y), labels_fz_y_size - 1)

            if (
                labels_fz[i, j] != labels_fz[x, y]
                and labels_fz[i, j] not in G[labels_fz[x, y]]
            ):

                rij = im[labels_fz == labels_fz[i, j]]
                rxy = im[labels_fz == labels_fz[x, y]]

                lenrij = len(rij)
                lenrxy = len(rxy)

                diff = np.linalg.norm(np.mean(rij) - np.mean(rxy))

                G.add_edge(
                    labels_fz[i, j],
                    labels_fz[x, y],
                    weight=np.exp(-diff ** 2 / 255),
                )
# nx.draw(G, ax=ax3, with_labels=True, font_weight="bold")

A = nx.adjacency_matrix(G).toarray()

# 理论上可以用拉普拉斯矩阵的特征值间的剧烈变化来得到最适合的K
# L = nx.laplacian_matrix(G).toarray()
# eigvals = np.linalg.eigvals(L)

K = 5

β = 10
similarity = np.exp(-β * A / A.std())

labels_sc = spectral_clustering(
    similarity,
    n_clusters=K,
    assign_labels="kmeans",
    eigen_solver="amg",
    random_state=42,
)

labels_final = np.zeros(labels_fz.shape, labels_fz.dtype)
for x in range(labels_final.shape[0]):
    for y in range(labels_final.shape[1]):
        i = labels_fz[x, y]
        labels_final[x, y] = labels_sc[i]

ax3.imshow(seg.mark_boundaries(im, labels_final))

ax4.imshow(color.label2rgb(labels_final, im, kind="avg"))

## vanilla spectral cluster(very very slow):
# from scipy.ndimage.filters import gaussian_filter
# from sklearn.feature_extraction import image
# smoothened_im = gaussian_filter(im, sigma=2)
# graph = image.img_to_graph(smoothened_im)
# beta = 10
# eps = 1e-6
# graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
# labels_sc1 = spectral_clustering(
#     graph,
#     n_clusters=K,
#     assign_labels="kmeans",
#     eigen_solver="amg",
#     random_state=42,
# )
# labels_sc1 = labels_sc1.reshape(smoothened_im.shape)
# ax1.set_title("Original + Spectral Clustering")
# ax1.imshow(color.label2rgb(labels_sc1, smoothened_im, kind='avg'))

plt.imsave(outputfile, color.label2rgb(labels_final, im, kind="avg"))
plt.show()