import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

class Heatmapper:
    def __init__(self, shape = (1920, 1080)):
        self.history = np.zeros(shape, dtype=np.int)

    def add_point(self, point):
        self.history[point[0], point[1]] += 1

    def generate_heatmap(self):
        img = self.history / self.history.argmax()
        img = ndimage.filters.gaussian_filter(img, sigma=20)
        plt.imshow(img)
        plt.show()
        return img


if __name__ == "__main__":
    hm = Heatmapper()
    for i in range(500, 900):
        for j in range(200, 400):
            hm.add_point((i, j))

    hm.generate_heatmap()

