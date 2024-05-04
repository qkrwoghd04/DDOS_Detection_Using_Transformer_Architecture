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

## Result
### Information
|Column |	Explanation|
|-------|------------|
|Model|	The name of the model being evaluated.|
|Time|	The time taken to evaluate the model.|
|Packets|	The total number of packets processed during evaluation.|
|Samples|	The total number of samples (flows) used for evaluation.|
|DDOS%|	The percentage of DDoS samples in the dataset.|
|Accuracy|	The overall accuracy of the model in classifying samples as either benign or DDoS.|
|F1Score|	The F1 score, a balanced measure of precision and recall.|
|TPR|	The true positive rate (sensitivity), the proportion of actual DDoS samples correctly identified as DDoS.|
|FPR|	The false positive rate, the proportion of benign samples incorrectly identified as DDoS.|
|TNR|	The true negative rate (specificity), the proportion of actual benign samples correctly identified as benign.|
|FNR|	The false negative rate, the proportion of actual DDoS samples incorrectly identified as benign.|
|Source|	The source of the dataset or evaluation results.|

---
## It can be observed that the GRU model outperforms the conventional CNN model
### Reason
GRU models are a type of recurrent neural network (RNN) that are powerful at processing time-series data. Since DDoS attacks involve network traffic data with patterns that change over time, GRU models are well-suited for detecting these attacks. CNN models may have lower performance in DDoS detection because they have relatively weaker time-series data processing capabilities.
|Model|Time|Packets|Samples|DDOS%|Accuracy|F1Score|TPR|FPR|TNR|FNR|Source|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|Existing CNN|0.052|3748|906|0.531|0.9404|0.9408|0.9954|0.1095|0.8905|0.0046|10t-10n-DOS2019-dataset-test.hdf5|
|Implemented GRU|0.038|3748|906|0.458|**0.9603**|0.9574|0.9397|0.0211|0.9789|0.0603|10t-10n-DOS2019-dataset-test.hdf5|
|Implemented LSTM|0.044|3748|906|0.398|0.9139|0.9015|0.8283|0.0084|0.9916|0.1717|10t-10n-DOS2019-dataset-test.hdf5|


---
## Dataset
The dataset employed labeled pcap files containing traffic traces, and preprocessing functionalities were implemented through the ddos_dataset_parser.py script
[Dataset Link](https://www.unb.ca/cic/datasets/ddos-2019.html)
<img width="958" alt="dataset" src="https://github.com/qkrwoghd04/DDOS_Detection_Using_Various_Architecture/assets/122519801/7e47050d-199d-435a-b56e-0ba1554e1005">

---
## Reference
https://github.com/Abhirambs-08/DDos-Detection-Using-Machine-Learning-Algorithms-Python/blob/main/Dataset%20and%20code/DDoS%20Detection%20using%20ML.ipynb<br>
https://github.com/doriguzzi/lucid-ddos<br>
https://github.com/mvoassis/CIC-DDoS2019-DeepLearning/blob/main/GRU-DDoS2019.ipynb<br>
