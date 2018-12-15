import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pylab import *


class Heatmapper:
    def __init__(self, shape = (1080, 1920), frame_every_points = 20, points_per_frame = 100):
        self.full_history = np.zeros(shape, dtype=np.int)
        self.history = list()
        self.frames = list()
        self.frame_every_points = frame_every_points

    def add_point(self, point):
        self.full_history[point[0], point[1]] += 1
        self.history.append(point)

    def generate_fulll_history_heatmap(self):
        img = self.full_history / self.full_history.argmax()
        img = ndimage.filters.gaussian_filter(img, sigma=20)
        plt.imshow(img)
        plt.show()
        return img

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_gameframe_for_frame(self, frame_n):
        frame = self.frames[frame_n//self.frame_every_points]
        return frame

    def generate_heatmap_for_frame(self, frame_n):
        hm_start = frame_n #- self.frame_every_points
        # if hm_start < 0:
        #     hm_start = 0
        hm_end = frame_n + self.frame_every_points
        if hm_end >= len(self.full_history):
            hm_end = len(self.full_history) - 1
        elif hm_end == self.frame_every_points:
            hm_end = self.frame_every_points * 2

        heatmap = np.zeros(self.full_history.shape, dtype=np.float)
        for i in range(hm_start, hm_end +1):
            p = self.history[i]
            heatmap[p[0], p[1]] += 1
        img = ndimage.filters.gaussian_filter(heatmap, sigma=40)*500

        return img

    def generate_video(self):
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax_game = fig.add_subplot(212)
        ax_game.set_aspect('equal')
        ax_game.get_xaxis().set_visible(False)
        ax_game.get_yaxis().set_visible(False)

        im = ax.imshow(self.generate_heatmap_for_frame(0))
        im.set_clim([0, 1])
        imm = ax_game.imshow(self.get_gameframe_for_frame(0))
        imm.set_clim([0, 1])
        fig.set_size_inches([5, 5])
        # Setting tight layout for pyplot
        tight_layout()

        def update_img(n):
            tmp = self.generate_heatmap_for_frame(n)
            tmp_gm = self.get_gameframe_for_frame(n)
            im.set_data(tmp)
            imm.set_data(tmp_gm)
            return [im, imm]

        #legend(loc=0)
        ani = animation.FuncAnimation(fig, update_img, 200, interval=5)
        writer = animation.writers['ffmpeg'](fps=60)

        ani.save('heatmap.mp4', writer=writer, dpi=100)
        return ani


if __name__ == "__main__":
    hm = Heatmapper(frame_every_points=50)
    mu, sigma = 0.2, 0.3
    xx = np.array(((np.random.normal(mu, sigma, 10000) * 400) + 400), dtype=np.uint16)
    yy = np.array(((np.random.normal(mu, sigma, 10000) * 600) + 850), dtype=np.uint16)
    print(xx.shape)
    for i in range(len(xx)):
        hm.add_point((xx[i], yy[i]))
    for i in range(len(xx) // 50):
        f = np.random.rand(1080, 1920)
        hm.add_frame(f)
    # hm.generate_fulll_history_heatmap()
    hm.generate_video()

