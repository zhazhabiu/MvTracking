import os
import gc
import time
import math
import numpy as np
import pandas as pd
import _winapi
_winapi.SYNCHRONIZE = 1
import sys
sys.path.append(os.getcwd())
from MvTracker import MultiViewTracker, unwrap_compress_axis
from concurrent.futures import ProcessPoolExecutor, as_completed
from searching import seed_generation, DFS, watershed, Meanshift2D
from trajectory_merge import merge_planar_track
# Multi-Process
Processpool = ProcessPoolExecutor(max_workers=3)

def load_dataset(path_dataset, sequence, fname):
    print(f'Reading from {path_dataset}/{sequence}/{fname}')
    if sequence=='star_tracking': # 1ms
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=",", header=None)  
        events.columns = ['timestamp', 'y', 'x', 'polarity'] 
        events_set = events.to_numpy()
        events_set = events_set[:, [0, 2, 1, 3]] # [t, y, x, p] -> [t, x, y, p]
        take_id = np.logical_and(np.logical_and(np.logical_and(events_set[:, 1] >= 0, \
                                                               events_set[:, 2] >= 0), \
                                                               events_set[:, 1] < 240), \
                                                               events_set[:, 2] < 180)
        events_set = events_set[take_id]
        print("Time duration of the sequence: {} s".format(events_set[-1, 0]*1e-3))
        events_set[:, 0] *= 1e+3 # us
        t_intervals = None
        print("Events total count: ", len(events_set))
        del take_id
    else: # txt
        events = pd.read_csv(
            '{}/{}/{}'.format(path_dataset, sequence, fname), sep=" ", header=None)  
        events.columns = ['t', 'x', 'y', 'p']  
        events_set = events.to_numpy()
        print("Events total count: ", len(events_set))
        print("Time duration of the sequence: {} s".format(events_set[-1, 0] - events_set[0, 0]))
        events_set[:, 0] *= 1e+6    # s -> us
        t_intervals = None
    events_set = events_set.astype(np.int64)
    return events_set, t_intervals


class TrackPerPlanar():
    def __init__(self, planar_id, region_dots_thres, approach='RG', param=None, save_dir='./three-palanar-show'):
        self.planar_id = planar_id
        self.region_dots_thres = region_dots_thres
        self.save_dir = save_dir
        self.approach = approach
        if approach == 'RG':
            # neighborhood
            self.param = 5 if param is None else param
        elif approach == 'WS':
            # threshold
            self.param = 0.7 if param is None else param
        elif approach == 'MS':
            # bandwidth
            self.param = 25 if param is None else param
        else:
            print("Not implemented.")
            raise

    def detect_(self, events, planar, Amplitude, realigned_T, shrink):
        if self.approach == 'RG':
            '''Region Growing'''
            seeds = seed_generation(Amplitude)
            planar_regions = DFS(seeds,
                                planar,
                                neighbors=self.param)
        elif self.approach == 'WS':
            '''Watershed'''
            planar_regions = watershed(planar, Amplitude, th=self.param)
        else:
            '''Meanshift'''
            planar_regions = Meanshift2D(planar, bandwidth=self.param)
        
        if len(planar_regions)==0:
            return self.planar_id, []
        if self.planar_id != 0 and len(planar_regions) > 1:
            planar_regions = merge_planar_track(planar_regions, self.planar_id)
            
        """unwrap the searched regions"""
        planar_record_region = []
        cnt = 0
                
        for region_ in planar_regions:
            cnt += 1
            origin_coor_save = unwrap_compress_axis(self.planar_id, region_, \
                                                        events[:, [(self.planar_id+2)%3, (self.planar_id+1)%3]], \
                                                        events[:, self.planar_id], events[:, -1],\
                                                        realigned_T, shrink)
            if len(origin_coor_save) <= self.region_dots_thres:
                continue
            # 0=j: y, x, t, p = y, x, yy 2, xx 1, tt 
            # 1=j: t, y, x, p = t, y, tt 0, yy 2, xx
            # 2=j: x, t, y, p = x, t, xx 1, tt 0, yy
            t_x_y_p = [(self.planar_id+2)%3, (self.planar_id+1)%3, self.planar_id, 3]
            origin_coor_save = origin_coor_save[:, t_x_y_p] # convert to [t, x, y, p]
            planar_record_region.append(origin_coor_save)
        del planar, planar_regions, region_, events, Amplitude
        return self.planar_id, planar_record_region


if __name__ == '__main__':
    theta = 30
    dataset_path, sequence, fname, image_size, dn = 'dataset', '', 'events.txt', (180, 240), 1800 # number of events
    # dataset_path, sequence, fname, image_size, dt = 'dataset', 'star_tracking', 'Sequence1.csv', (180, 240), 40000 # us
    
    events_set, t_intervals = load_dataset(dataset_path, sequence, fname)
    
    if sequence=='star_tracking':
        save_dir = './output' + sequence + fname[:-4]
    else:
        save_dir = './output' + sequence
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'Tracks'))
    else:
        files = os.listdir(os.path.join(save_dir, 'Tracks'))
        print('There are tracks')
            
    print(save_dir)
    
    L = MultiViewTracker(events_set, image_size, t_axissize=200, 
                            save_dir=save_dir, dt=None, dn=dn, 
                            theta=math.pi/180*theta, b_search=9,
                            approach='RG', param=5)
    planar_dets = [TrackPerPlanar(i, L.region_dots_thres, L.approach, L.param, save_dir) for i in range(3)] # XOY YOT TOX
    for i, start_ind in enumerate(L.intervals[:-1]):
        start_time = time.time()
        end_ind = L.intervals[i+1]
        indices = events_set[start_ind:end_ind, :] # [t, x, y, p]
        inputs0, inputs1 = L.get_views(indices)
        """each planar dealing""" 
        deals = []
        for j in range(3):
            deal = Processpool.submit(planar_dets[j].detect_, indices, inputs0[j], inputs1[j], L.realigned_T, L.shrinks)
            deals.append(deal)
        
        deal_res = []
        for deal in as_completed(deals):
            deal_res.append(deal.result())
            del deal
        
        del indices, inputs0, inputs1, deals
        gc.collect()
        """fuse three planar"""
        L.planar_merge(deal_res, i)
        del deal_res
        print(f'Batch {i}  Time: {time.time() - start_time}   {len(L.trajectories)}')
        gc.collect()