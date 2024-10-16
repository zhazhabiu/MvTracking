# coding=utf-8
'''多线程'''
import cv2
import time
import copy
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from searching import DFS, watershed, Meanshift2D
import trajectory_merge
import gc

# Multi-Process
Processpool = ProcessPoolExecutor(max_workers=3)

def get_realigned_coord(t, realigned_T, shrink):
    return ((np.nonzero(np.equal(t[:, None], realigned_T[None, :]))[1])*shrink).astype(np.int16)
    
def GradientAmp(im):
    sobelx = cv2.Sobel(im, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(im, cv2.CV_16S, 0, 1, ksize=3)
    scharrx = cv2.Scharr(im,cv2.CV_16S, 1, 0)
    scharry = cv2.Scharr(im,cv2.CV_16S, 0, 1)
    gra_x = (scharrx - sobelx) / 2
    gra_y = (scharry - sobely) / 2
    Amplitude = np.sqrt(np.power(gra_x, 2) + np.power(gra_y, 2))
    del sobelx, sobely, scharrx, scharry, gra_x, gra_y
    return Amplitude

class TrackPerPlanar():
    def __init__(self, planar_id, region_dots_thres, save_dir='./three-palanar-show'):
        self.planar_id = planar_id
        self.region_dots_thres = region_dots_thres
        self.save_dir = save_dir
    
    def seed_generation(self, amp):
        """Initialize to number of points clusters
        Args:
            amp (array): map of gradient amplitute
        Returns:
            seeds (List): [tensor(size(1, 2)), tensor(size(1, 2)),...] regard each point is independent
        """
        nonzeros_index = np.nonzero(amp)
        seed_index = np.argsort(-amp[nonzeros_index])
        seeds = [(nonzeros_index[0][i], nonzeros_index[1][i]) for i in seed_index]
        del nonzeros_index, seed_index
        return seeds

    def unwrap_compress_axis(self, ab, ab_origin, c_origin, p_origin, realigned_T, shrink):
        ab_origin_ = ab_origin.copy()
        if self.planar_id != 0:
            # recover timestamps
            t_axis = (self.planar_id+2)%3
            ab_origin_[:, t_axis] = get_realigned_coord(ab_origin_[:, t_axis], realigned_T, shrink) #

        event_ids = []
        ab = np.array_split(ab, 10, axis=0)
        for aabb in ab:
            event_ids.append(np.nonzero(np.abs(aabb[None, :, :] - ab_origin_[:, None, :]).sum(-1)==0)[0])
        event_ids = np.unique(np.concatenate(event_ids, 0))
        save = np.stack((ab_origin[event_ids, 0], ab_origin[event_ids, 1], c_origin[event_ids], p_origin[event_ids]), axis=1)
        del event_ids, ab_origin_, ab
        return save
    
    def detect_(self, events, planar, Amplitude, realigned_T, shrink):
        start_t = time.time()
        '''Region Growing'''
        seeds = self.seed_generation(Amplitude)
        planar_regions = DFS(seeds,
                            planar,
                            neighbors=5)
        '''Watershed'''
        # planar_regions = watershed(planar, Amplitude, th=0.7)
        '''Meanshift'''
        # planar_regions = Meanshift2D(planar, bandwidth=25)
        
        if len(planar_regions)==0:
            return self.planar_id, []
        if self.planar_id != 0 and len(planar_regions) > 1:
            planar_regions = trajectory_merge.merge_planar_track(planar_regions, self.planar_id)
            
        """unwrap the searched regions"""
        planar_record_region = []
        cnt = 0
        spend_t = 0 
                
        for region_ in planar_regions:
            cnt += 1
            start_t = time.time()
            origin_coor_save = self.unwrap_compress_axis(region_, \
                                                        events[:, [(self.planar_id+2)%3, (self.planar_id+1)%3]], \
                                                        events[:, self.planar_id], events[:, -1],\
                                                        realigned_T, shrink)
            if len(origin_coor_save) <= self.region_dots_thres:
                continue
            spend_t += time.time() - start_t
            # 0=j: y, x, t, p = y, x, yy 2, xx 1, tt 
            # 1=j: t, y, x, p = t, y, tt 0, yy 2, xx
            # 2=j: x, t, y, p = x, t, xx 1, tt 0, yy
            t_x_y_p = [(self.planar_id+2)%3, (self.planar_id+1)%3, self.planar_id, 3]
            origin_coor_save = origin_coor_save[:, t_x_y_p] # convert to [t, x, y, p]
            planar_record_region.append(origin_coor_save)
        del planar, planar_regions, region_, events, Amplitude
        return self.planar_id, planar_record_region

class MultiViewTracker():
    def __init__(self, events_stream, XOY_size, t_axissize=500, 
                 save_dir='./three-palanar-show',
                 dt=2000, dn=2000, theta=math.pi/10, b_search=9,
                 region_dots_thres=20):
        super(MultiViewTracker, self).__init__()
        self.events = events_stream.copy() # [t, x, y, p]
        self.batch_search = b_search
        self.region_dots_thres = region_dots_thres
        self.save_dir = save_dir
        self.t_axissize = t_axissize
        self.shrinks = 1
        self.t_shape = 0
        # timestamp reset
        self.zero_timestamp = self.events[0, 0]
        self.events[:, 0] = self.events[:, 0] - self.events[0, 0]
        if dt is None:
            self.intervals = list(range(self.events[0, 0], self.events[-1, 0], dn))
        else:
            self.intervals = []
            for start_t in np.arange(self.events[0, 0], self.events[-1, 0], dt):
                start_ind = np.nonzero(self.events[:, 0] >= start_t)[0][0]
                self.intervals.append(start_ind)
        self.intervals.append(len(self.events)-1)
        self.theta = theta
        self.scale = 0.8
        self.XOY_size = XOY_size
        self.trajectories = []  # save the trajectories within batch_search time windows
        self.trajec_ids = []
        self.last_use = []
        self.realigned_T = None
        self.timestamp2alignedT = None
        self.dt = dt
        
    def integrate_to_planar(self, indices, planar_id):
        """
        A event representation -- Three Planar Representation
        Args:
            indices (np.array): events stream
            planar_id (int): index of projection planar
        Returns:
            planar
        """ 
        if planar_id == 1:
            # Timestamp-realigned
            self.realigned_T = np.unique(indices[:, 0])
            if self.t_axissize < len(self.realigned_T):
                self.shrinks = self.t_axissize / len(self.realigned_T)
                self.t_shape = self.t_axissize + 1 
            else:
                self.shrinks = 1
                self.t_shape = len(self.realigned_T)
            planar_size = (self.t_shape, self.XOY_size[0])
            planar = np.zeros(planar_size, dtype=np.float32)
            self.timestamp2alignedT = get_realigned_coord(indices[:, 0], self.realigned_T, self.shrinks)
            tmp = np.stack((self.timestamp2alignedT, indices[:, 2]), axis=1)
            coords, vals = np.unique(tmp, axis=0, return_counts=True)
            planar[coords[:, 0], coords[:, 1]] = vals
            del tmp, coords, vals
        elif planar_id == 2:
            planar_size = (self.XOY_size[1], self.t_shape)
            planar = np.zeros(planar_size, dtype=np.float32)
            tmp = np.stack((indices[:, 1], self.timestamp2alignedT), axis=1)
            coords, vals = np.unique(tmp, axis=0, return_counts=True)
            planar[coords[:, 0], coords[:, 1]] = vals
            del self.timestamp2alignedT, tmp, coords, vals
        else:
            planar_size = self.XOY_size
            planar = np.zeros(planar_size, dtype=np.float32)
            coords, vals = np.unique(indices[:, [2, 1]], axis=0, return_counts=True)
            planar[coords[:, 0], coords[:, 1]] = vals
            del coords, vals
        """mini-gaussian pyramid enhance"""
        planar = self.gaussian_pyramid(planar, planar_size)
        Amplitude = GradientAmp(planar)
        return planar, Amplitude 

    def gaussian_pyramid(self, input, planar_size):
        out0 = cv2.GaussianBlur(input, (3, 3), self.scale/0.5)
        # out0  = input
        size0 = (int(planar_size[1]*self.scale), int(planar_size[0]*self.scale))
        ''' Attention !!! cv2 operation use [w, h], and numpy use [h, w], example numpy.shape represents [h, w]'''
        out0 = cv2.resize(out0, size0, interpolation=cv2.INTER_LINEAR)
        out0 = cv2.resize(out0, planar_size[::-1], interpolation=cv2.INTER_LINEAR) + input
        return out0
    
    def merge_two_planar(self, o1, o2):
        merges = []
        for track1 in o1:
            for track2 in o2:
                tmp = np.concatenate((track1, track2)) # [N, 4]
                tmp, count = np.unique(tmp, return_counts=True, axis=0)
                tmp = tmp[count>1]
                del count
                if len(tmp) < 2:
                    continue
                merges.append(tmp)
        return merges
    
    def planar_merge(self, deal_res, batch):
        planars = [None]*3
        for i in range(3):
            pid = deal_res[i][0]
            planars[pid] = deal_res[i][1]
        
        XOY, YOT, TOX = planars
        mergeXOY_YOT = self.merge_two_planar(XOY, YOT)
        del XOY, YOT
        cur_tracks = self.merge_two_planar(mergeXOY_YOT, TOX)
        del mergeXOY_YOT, TOX
        
        cur_tracks = trajectory_merge.merge_track_inbatch(cur_tracks, self.realigned_T, self.shrinks, self.theta, self.region_dots_thres)
        
        if len(cur_tracks) == 0:
            return
        cur_tracks_rt = copy.deepcopy(cur_tracks)
        match_ids = trajectory_merge.merge_twindow_track(self.trajectories, 
                                        cur_tracks_rt, 
                                        self.realigned_T, 
                                        self.shrinks,
                                        self.theta)
        '''update self.trajectories and self.trajec_ids'''
        max_id = -1 if len(self.trajec_ids)==0 else max(self.trajec_ids) 
        cur_ids = []
        for i, j in enumerate(match_ids):
            if j >= 0:
                self.trajectories[j] = cur_tracks_rt[i]
                self.last_use[j] = batch
                cur_ids.append(self.trajec_ids[j])
            else:
                # deal with new tracks
                self.trajectories.append(cur_tracks_rt[i])
                max_id += 1
                self.trajec_ids.append(max_id)
                self.last_use.append(batch)
                cur_ids.append(max_id)
        trajectories = []
        new_ids = []
        new_last_use = []
        for i, j in enumerate(self.last_use):
            if batch < self.batch_search or j >= batch-self.batch_search+1:
                trajectories.append(self.trajectories[i])
                new_ids.append(self.trajec_ids[i])
                new_last_use.append(j)
        self.trajectories = trajectories
        self.trajec_ids = new_ids
        self.last_use = new_last_use
        '''show_3d_events(cur_tracks, cur_ids, batch, self.save_dir)'''
        for id, trajectory in zip(cur_ids, cur_tracks):
            trajectory = trajectory[np.argsort(trajectory[:, 0])]
            trajectory[:, 0] = trajectory[:, 0] + self.zero_timestamp # timestamp recovery
            with open(f'{self.save_dir}/Tracks/{id}.txt', 'a+') as f:
                np.savetxt(f, np.c_[trajectory], fmt='%d', delimiter=',') # us, x, y, p
            del trajectory
        del pid, planars, cur_tracks, deal_res, trajectories, new_ids, match_ids, cur_tracks_rt, new_last_use, max_id
    
    def detect(self):
        """
        Detect the trajectory within batches
        """
        planar_dets = [TrackPerPlanar(i, self.region_dots_thres, self.save_dir) for i in range(3)] # XOY YOT TOX
        for i, start_ind in enumerate(self.intervals[:-1]):
            end_ind = self.intervals[i+1]
            indices = self.events[start_ind:end_ind, :] # [t, x, y, p]
            XOY, XOY_amp = self.integrate_to_planar(indices, 0)
            denoise_indices = indices
            YOT, YOT_amp = self.integrate_to_planar(denoise_indices, 1)
            TOX, TOX_amp = self.integrate_to_planar(denoise_indices, 2)
            start_time = time.time()
            inputs0 = [XOY, YOT, TOX]
            inputs1 = [XOY_amp, YOT_amp, TOX_amp]
            """each planar dealing""" 
            deals = []
            for j in range(3):
                deal = Processpool.submit(planar_dets[j].detect_, denoise_indices, inputs0[j], inputs1[j], self.realigned_T, self.shrinks)
                deals.append(deal)
            
            deal_res = []
            for deal in as_completed(deals):
                deal_res.append(deal.result())
                del deal
            
            del indices, denoise_indices, inputs0, inputs1, XOY, YOT, TOX, XOY_amp, YOT_amp, TOX_amp, deals
            gc.collect()
            """fuse three planar"""
            self.planar_merge(deal_res, i)
            del deal_res
            print(f'Batch {i}  Time: {time.time() - start_time}   {len(self.trajectories)}')
            gc.collect()
