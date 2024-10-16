import numpy as np
import numba as nb
from sklearn.cluster import MeanShift
import cv2
import _winapi
_winapi.SYNCHRONIZE = 1


@nb.jit(nopython=True)
def get_neighbors(point, p=3, visited=None):
    r = (p - 1)//2
    neighbord = []
    # border condition
    h, w = visited.shape
    for i in range(-r, r+1, 1):
        for j in range(-r, r+1, 1):
            y, x = point[0]+i, point[1]+j
            pt = (y, x)
            if visited is not None:
                if x >= 0 and y >= 0 and x < w and y < h and visited[y][x]:
                    neighbord.append(pt)
                continue
            neighbord.append(pt)
    return neighbord

def DFS(seeds, planar, neighbors=3):
    """Depth First Search
    Args:
        seeds ([seed, seed, ...])
        planar (array): used as a record of whether the point has been visited and also a constraint
        neighbors (int, optional): Defaults to 3
        region_thres(int, optional): Defaults to 30
    Returns:
        regions (List): each region size should be restricted to (num, 2)
    """
    regions = [] 
    for seed in seeds:
        # tranverse seeds of each region
        region = []
        region_seeds = [seed]
        while len(region_seeds) > 0:
            seed = region_seeds[0]
            region_seeds.pop(0)
            if planar[seed[0]][seed[1]]:
                region.append(seed)
                planar[seed[0]][seed[1]] = 0
                seed_list = get_neighbors(seed, neighbors, planar)
                if len(seed_list) > 0:
                    region_seeds.extend(seed_list)
        if len(region)==0:
            continue
        region = np.stack(region, 0) # Size([num, 2])
        regions.append(region)
    return regions

def Meanshift2D(im, bandwidth=2):
    # convert im to data
    sample = np.transpose(np.nonzero(im))
    clustering  = MeanShift(bandwidth=bandwidth).fit(sample)
    n_clusters, _ = clustering.cluster_centers_.shape
    regions = []
    for i in range(n_clusters):
        regions.append(sample[np.nonzero(clustering.labels_==i)[0]])
    return regions

def watershed(im, gradient, th = 0.5):
    '''seed watershed'''
    im = (((im - im.min())/(im.max() - im.min()))*255).astype(np.uint8)
    gradient = (((gradient - gradient.min())/(gradient.max() - gradient.min()))*255).astype(np.uint8)
    markers = gradient.copy()
    markers[markers > 255*th] = 0
    ret, markers = cv2.connectedComponents(markers)
    markers = markers + 1 # 标记背景像素点为0，非背景像素点从1开始累加分别标记
    
    im = np.repeat(im[:, :, None], 3, axis=2).astype(np.uint8)
    labels = cv2.watershed(im, markers) #基于梯度的分水岭算法
    '''Gradient watershed'''
    '''
    from skimage import segmentation
    from scipy import ndimage as ndi
    im = (((im - im.min())/(im.max() - im.min()))*255).astype(np.uint8)
    gradient = (((gradient - gradient.min())/(gradient.max() - gradient.min()))*255).astype(np.uint8)
    markers = gradient < 255*th
    markers = ndi.label(markers)[0]
    labels = segmentation.watershed(gradient, markers, mask=im) #基于梯度的分水岭算法
    '''
    n_labels = labels.max()
    regions = []
    for i in range(2, n_labels+1):
        regions.append(np.transpose(np.nonzero(labels==i)))
    return regions
