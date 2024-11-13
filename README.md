# MvT: Multi-view Tracking
This is the official implementation of《MvT: Event-based Multi-view Projection for
Multiple Object Tracking》using Python.  


The demo is published.
Source code will be published when the paper is accepted.

## Examples on Experiments 
ECDataset
![image](./visualize/fig9.jpg)

Small Object Dataset
![image](./visualize/fig12.jpg)


## Prerequisites 
```
python==3.9.0    
numpy==1.24.0  
opencv-python==4.10  
opencv-contrib-python  
scikit-learn==1.5.1  
pandas==2.2.2  
numba
```

## OS
```
Windows 
```  

## Run Demo
```Python
python demo/main.cmd  
 # Please run in the environment with Prerequisites
```

## Run source code (not published yet)
```Python
python src/main.py  
```

## Data preparation for source code (not published yet)
Data should be put in *'./dataset/{name_of_dataset}/events.txt(csv)'*, or you can change the file path in *src/main.py*.


## Tracking results
All the trajectories will be saved in *'./output/{name_of_dataset}/Tracks'*.


## Visualize trackings
```Python
python visualize/show.py  
```