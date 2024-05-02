# DDOS_Detection_Using_Various_Architecture

This project aims to learn the characteristics of DDoS attack traffic and normal traffic by using various model architectures for DDoS attack detection, and reduce both processing cost and attack detection time by using a small number of layers.

## Setting Environment(Window based)
```
conda create -n python311 python=3.11
conda activate python39
```
Now you can see the conda environment on vscode
<img width="665" alt="1" src="https://github.com/qkrwoghd04/DDOS_Detection_Using_Various_Architecture/assets/122519801/95bdf2fe-aa54-4ef1-83d6-885c6c416550">
```
(python311)pip install tensorflow==2.9.0
(python311)pip install scikit-learn h5py pyshark protobuf==3.19.6
```

## Data Loading
```
python ddos_datasest_parser.py --dataset_type DOS2019 --dataset_folder ./Dataset/ --packets_per_flow 10 --dataset_id DOS2019 --traffic_type all --time_window 10
```

## Data Preprocessing
```
python ddos_dataset_parser.py --preprocess_folder Dataset
```

## Training model
```
python (name of model ython file).py --train Dataset  
```

## Testing
```
python (name of model python file).py --predict Dataset   
```
