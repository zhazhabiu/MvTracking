import os
import numpy as np
import pandas as pd
from MvTracker import MultiViewTracker

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

if __name__ == '__main__':
    theta = 30
    dataset_path, sequence, fname, image_size, dn = 'dataset', 'shapes_translation', 'events.txt', (180, 240), 1800 # number of events
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
            
    import math
    print(save_dir)
    L = MultiViewTracker(events_set, image_size, t_axissize=200, 
                            save_dir=save_dir, dt=None, dn=dn, 
                            theta=math.pi/180*theta, b_search=9,
                            approach='RG', param=5)
    L.detect()