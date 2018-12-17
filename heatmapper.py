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

    def generate_heatmap_for_frame(self, frame_n, blur):
        hm_start = frame_n -100#- self.frame_every_points
        if hm_start < 0:
            hm_start = 0
        hm_end = frame_n + self.frame_every_points
        if hm_end >= len(self.history):
            hm_end = len(self.history) - 1
        elif hm_end == self.frame_every_points:
            hm_end = self.frame_every_points * 2

        # print(blur_kernel)

        print(blur.shape)
        heatmap = np.zeros(self.full_history.shape, dtype=np.float)

        put_indexes = self.history[hm_start:hm_end, :]
        put_indexes_l = put_indexes - 30
        put_indexes_h = put_indexes + 30
        # put_indexes_hl = np.vstack([put_indexes_l[:, 0].T, put_indexes_h[:, 0].T, put_indexes_l[:, 1].T, put_indexes_h[:, 1].T]).T
        # heatmap[put_indexes_l[:, 0]:put_indexes_h[:, 0], put_indexes_l[:, 1]:put_indexes_h[:, 1]] += blur
        # put_indexes_hl = np.zeros((hm_end - hm_start, 2, 60), dtype=np.uint16)
        # for i in range(len(put_indexes)):
        #     put_indexes_hl[i, 0, :] = np.arange(put_indexes[i, 0]-30, put_indexes[i, 0] + 30)
        #     put_indexes_hl[i, 1, :] = np.arange(put_indexes[i, 1]-30, put_indexes[i, 1] + 30)
        # # heatmap = np.tile(heatmap, (len(put_indexes), 1, 1))
        # heatmap[:, put_indexes_hl[:, 0, :], put_indexes_hl[:, 1, :]] += np.array([blur]*len(put_indexes))
        # heatmap = np.sum(heatmap, axis=0)
        # print(heatmap)
        for i in range(len(put_indexes)):
            heatmap[put_indexes_l[i, 0]:put_indexes_h[i, 0], put_indexes_l[i, 1]:put_indexes_h[i, 1]] += blur
        return heatmap

    def generate_video_process(self):
        stt = time.time()
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax_game = fig.add_subplot(212)
        ax_game.set_aspect('equal')
        ax_game.get_xaxis().set_visible(False)
        ax_game.get_yaxis().set_visible(False)

        self.history = np.array(self.history)
        self.history = np.clip(self.history, 31, 1887)
        self.history[:, 0] = np.clip(self.history[:, 0], 31, 1047)

        blur_kernel = self.gkern(60, 10) * 300
        blur = np.array(blur_kernel)

        im = ax.imshow(self.generate_heatmap_for_frame(0, blur))
        im.set_clim([0, 1])
        game_frame = np.array(self.get_gameframe_for_frame(0))
        game_frame[:, 0], game_frame[:, 2] = game_frame[:, 2], game_frame[:, 0].copy()

        imm = ax_game.imshow(game_frame)
        imm.set_clim([0, 1])
        fig.set_size_inches([8, 8])
        # Setting tight layout for pyplot
        tight_layout()

        def update_img(n):
            tmp = self.generate_heatmap_for_frame(n, blur)
            tmp_gm = np.array(self.get_gameframe_for_frame(n))
            tmp_gm[:, 0], tmp_gm[:, 2] = tmp_gm[:, 2], tmp_gm[:, 0].copy()
            im.set_data(tmp)
            imm.set_data(tmp_gm)
            return [im, imm]

        # legend(loc=0)
        ani = animation.FuncAnimation(fig, update_img, len(self.history) - 3, interval=1)
        writer = animation.writers['ffmpeg'](fps=50)

        ani.save('heatmap.mp4', writer=writer, dpi=45)
        print("Heatmap saved to heatmap.mp4")
        se = time.time()
        print(se - stt)
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

    hm.generate_video()


