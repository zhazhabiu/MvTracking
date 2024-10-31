import matplotlib.pyplot as plt
import numpy as np
import os

color = []
for a in np.linspace(0, 255, 5):
    for b in np.linspace(0, 255, 5):
        for c in np.linspace(100, 200, 2):
            color.append((a, b, c))
            
plt_color = ['b', 'lime',  'orange', 'gold', 'deepskyblue', 'darkgrey', 'brown', 'pink', 'tomato', 'orangered', 'purple',\
         'palegreen', 'slategrey', 'navy', 'darkorchid', 'turquoise', 'teal', 'saddlebrown','olive', 'greenyellow', \
         'crimson', 'c', 'g', 'k', 'm', 'r', 'w', 'y']

if __name__ == '__main__':
    plt.figure()
    img_size=(180, 240)
    ax = plt.axes(projection="3d")
    ax.set_ylim((0, img_size[1]))
    ax.set_zlim((0, img_size[0]))
    ax.set_xlabel('T')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    
    sequence = './visualize/'
    files = os.listdir(sequence)
    for cnt, f in enumerate(files):
        if f.endswith('txt'):
            track = np.loadtxt(sequence+'/'+f, dtype=np.int64, delimiter=',') # [t, x, y, p]
            ax.scatter3D(track[:, 0], track[:, 1], track[:, 2], c=plt_color[cnt%len(plt_color)], marker='.')
    plt.savefig(f'./visualize/viz.png')
    plt.close()