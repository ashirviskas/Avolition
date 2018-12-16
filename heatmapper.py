import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from pylab import *
import time
# import cv2
import scipy.stats as st
from multiprocessing import Process



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
        nn = frame_n//self.frame_every_points
        if nn >= len(self.frames):
            nn = len(self.frames) -1
        frame = self.frames[nn]
        return frame

    @staticmethod
    def gkern(l=5, sig=1.):
        """
        creates gaussian kernel with side length l and a sigma of sig
        """

        ax = np.arange(-l // 2 + 1., l // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sig ** 2))

        return kernel / np.sum(kernel)

    def generate_heatmap_for_frame(self, frame_n):
        hm_start = frame_n -100#- self.frame_every_points
        if hm_start < 0:
            hm_start = 0
        hm_end = frame_n + self.frame_every_points
        if hm_end >= len(self.full_history):
            hm_end = len(self.full_history) - 1
        elif hm_end == self.frame_every_points:
            hm_end = self.frame_every_points * 2
        blur_kernel = self.gkern(60, 10) * 300
        # print(blur_kernel)
        blur = np.array(blur_kernel)
        heatmap = np.zeros(self.full_history.shape, dtype=np.float)

        for i in range(hm_start, hm_end + 1):
            if i >= len(self.history):
                break
            p = self.history[i]
            x_p = p[1]
            y_p = p[0]
            if x_p < 31:
                x_p = 31
            if x_p > 1887:
                x_p = 1887
            if y_p < 31:
                y_p = 31
            if y_p > 1047:
                y_p = 1047
            print(p)
            print(x_p)
            print(y_p)
            heatmap[y_p - 30: y_p + 30, x_p - 30: x_p + 30] += blur
        # img = ndimage.filters.gaussian_filter(heatmap, sigma=15)*500
        # img = cv2.GaussianBlur(heatmap, (5, 5), 16)
        # img = self.vectorized_RBF_kernel(heatmap, 16)

        return heatmap

    def generate_video_process(self):
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
        game_frame = np.array(self.get_gameframe_for_frame(0))
        game_frame[:, 0], game_frame[:, 2] = game_frame[:, 2], game_frame[:, 0].copy()

        imm = ax_game.imshow(game_frame)
        imm.set_clim([0, 1])
        fig.set_size_inches([5, 5])
        # Setting tight layout for pyplot
        tight_layout()

        def update_img(n):
            tmp = self.generate_heatmap_for_frame(n)
            tmp_gm = np.array(self.get_gameframe_for_frame(n))
            tmp_gm[:, 0], tmp_gm[:, 2] = tmp_gm[:, 2], tmp_gm[:, 0].copy()
            im.set_data(tmp)
            imm.set_data(tmp_gm)
            return [im, imm]

        # legend(loc=0)
        ani = animation.FuncAnimation(fig, update_img, len(self.history) - 3, interval=1)
        writer = animation.writers['ffmpeg'](fps=60)

        ani.save('heatmap.mp4', writer=writer, dpi=100)
        print("Heatmap saved to heatmap.mp4")
        return ani

    def generate_video(self):
        p = Process(target=self.generate_video_process)
        p.start()


if __name__ == "__main__":
    hm = Heatmapper(frame_every_points=20)
    mu, sigma = 0.2, 0.3
    xx = np.array(((np.random.normal(mu, sigma, 100) * 400) + 400), dtype=np.uint16)
    yy = np.array(((np.random.normal(mu, sigma, 100) * 600) + 850), dtype=np.uint16)
    print(xx.shape)
    for i in range(len(xx)):
        hm.add_point((xx[i], yy[i]))
    for i in range((len(xx) // 20) + 1):
        f = np.random.rand(1080, 1920)
        hm.add_frame(f)
    # hm.generate_fulll_history_heatmap()
    stt = time.time()
    hm.generate_video()
    se = time.time()
    print(se-stt)

