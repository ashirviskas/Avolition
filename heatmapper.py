import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from pylab import *


class Heatmapper:
    def __init__(self, shape = (1080, 1920), frame_every_points = 20):
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

    def generate_video(self):



    def ani_frame(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        im = ax.imshow(rand(300, 300),cmap='gray',interpolation='nearest')
        im.set_clim([0,1])
        fig.set_size_inches([5,5])


        tight_layout()


        def update_img(n):
            tmp = rand(300,300)
            im.set_data(tmp)
            return im

        #legend(loc=0)
        ani = animation.FuncAnimation(fig,update_img,300,interval=30)
        writer = animation.writers['ffmpeg'](fps=30)

        ani.save('heatmap.mp4',writer=writer,dpi=dpi)
        return ani

if __name__ == "__main__":
    hm = Heatmapper()
    mu, sigma = 0.2, 0.3
    xx = np.array(((np.random.normal(mu, sigma, 1500) * 400) + 400), dtype=np.uint16)
    yy = np.array(((np.random.normal(mu, sigma, 1500) * 600) + 850), dtype=np.uint16)
    for i in range(len(xx)):
        hm.add_point((xx, yy))
    hm.generate_heatmap()

