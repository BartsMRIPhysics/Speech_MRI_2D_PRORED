# Speech_MRI_2D_PRORED

## Data

* Speech MRI Data - The data is available at [Zenodo](https://zenodo.org/records/10046815). The processed dataset in .npy form can be downloaded from [Google Drive](https://drive.google.com/file/d/1wT64P9YtIot7PrxMrnJRkXJ8T5sBSiWS/view?usp=sharing). Save the downloaded files to the 'Speech_MRI_2D_PRORED' folder path.

## Training
```
cd into Speech_MRI_2D_PRORED
```

For Speech MRI, run ``` CUDA_VISIBLE_DEVICES=0 python speech_train.py ```

## Testing
```
cd into Speech_MRI_2D_PRORED
```

For Speech MRI, run ``` CUDA_VISIBLE_DEVICES=0 python test_speech.py ```
