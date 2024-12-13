# Implement TLC method on MobileNet-V2
Here is an example of implementing TLC. This is the script that executes the TLC method for MobileNet-V2 model trained on different datasets.

To reproduce the experiment of TLC for MobileNet-V2 trained on CIFAR-10 in the paper by launching:
```bash
python TLC_MobileNetV2.py
```


### To use VLC for other datasets

It is easy to extend TLC to other datasets. Here below you can find the steps to follow.

* Before executing, ensure that the target dataset is already exist on your device. Then launch the experiment:
```bash
 python TLC_MobileNetV2.py  --dataset  'Target-dataset'  --DATA_DIR 'Your_Data_Location'  
```


## List of available datasets

- CIFAR-10
- Tiny-ImageNet-200
- PACS
- VLCS
- ImageNet-1000

