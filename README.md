# Speech_MRI_2D_PRORED

## Data

* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the 'Speech_MRI_2D_PRORED' folder path.
* ACDC - The data is available from [MICCAI2017 challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/miccai_results.html), or you can download the processed dataset from Google Drive of [MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view). Save the downloaded files to the 'Speech_MRI_2D_PRORED' folder.

## Training
```
cd into Speech_MRI_2D_PRORED
```

For Speech MRI, run 
```
CUDA_VISIBLE_DEVICES=0 python train_speech.py 
```

For ACDC, run
```
CUDA_VISIBLE_DEVICES=0 python train_ACDC.py 
```

## Testing
```
cd into Speech_MRI_2D_PRORED
```

For Speech MRI, please run the training code to get the split of the test data for each fold. Then run: 
``` 
CUDA_VISIBLE_DEVICES=0 python test_speech.py
```
For ACDC, run
```
CUDA_VISIBLE_DEVICES=0 python test_ACDC.py
```

## Reproducing the result
To test the dataset with trained model for :

### Speech
1. Create a new folder with name 'model_pth' to Speech_MRI_2D_PRORED. 
2. Download the trained model from  [Google Drive](https://drive.google.com/file/d/1y7rvY2ZcMsrV7Sg7D7WozxxZRo5-BPV8/view?usp=sharing). 
3. Unzip the downloaded folder and put the trained models in the folder './model_pth'. An example path to the saved models will be '.../Speech_MRI_2D_PRORED/model_pth/best_metric_model_0.pth'.
5. ```cd into Speech_MRI_2D_PRORED``` 
6. ```CUDA_VISIBLE_DEVICES=0 python test_speech.py``` 

### ACDC
1. Create a new folder with name 'model_pth' to Speech_MRI_2D_PRORED.
2. Download the trained model from [Google Drive](https://drive.google.com/file/d/1z_MZuVHQtG6Jmy4_8la0b-Th3eoqBBx0/view?usp=sharing).
3. Unzip the downloaded folder and put the trained model for ACDC in the folder './model_pth'.
4. ```cd into Speech_MRI_2D_PRORED```
5. ```CUDA_VISIBLE_DEVICES=0 python test_ACDC.py```
