# SATSURE_ASSIGNMENT_GEOSPATIAL

# Slides 
https://docs.google.com/presentation/d/1Phht4sepFN0L5BL3ASvaVB-DHCC6dR1WumF-X7-Uxxo/edit?usp=sharing

# Dataset
The Dataset is available for download [here](https://github.com/VSainteuf/pastis-benchmark). 

# Data Visualization
Data Visualization.ipynb file

# Implementation
## Requirements


### Python requirements

```setup
pip install -r requirements.txt
```

## U-TAE Spatio-Temporal Encoding Architecture
This architect shows a set of images is processed together using a convolutional encoder. A temporal encoder uses attention mechanisms to create sets of attention masks for each pixel, capturing temporal information, which are then distributed across different image resolutions to accommodate variations in image sizes. These masks are compressed into a single map for each resolution. Subsequently, a convolutional decoder is applied to generate features at different resolution levels. During this process, convolution operations exclusively consider spatial and channel dimensions, and strided convolutions are used to change spatial dimensions. Finally, the resulting feature maps are converted into RGB space to facilitate visual interpretation.


## Parcels-as-Points(PaPs) Module For Panoptic Segmentation
The Parcels-as-Points (PaPs) Module plays a pivotal role in the overall panoptic segmentation process. This module extends the capabilities of the system by treating the detected regions, or "parcels," as discrete points within the scene. By representing parcels as discrete points, it simplifies object localization, enhances semantic understanding, and improves the precision of pixel-level mask generation. For this segmentation, the heatmap identifies local peaks, which in turn correspond to M potential regions of interest to make predictions for several key attributes. The shape information is also combined with a global saliency map to enhance the precision of predicting pixel-level masks. These individual instance predictions are eventually integrated into a panoptic segmentation result.



## Results

Our model achieves the following performance on :

### PASTIS - Panoptic segmentation

The spatio-temporal encoder U-TAE combined with PaPs instance segmentation module achieves 40.4 Panoptic Quality (PQ) on PASTIS for panoptic segmentation. When replacing U-TAE with a convolutional LSTM the performance drops to 33.4 PQ.

| Model name         | SQ  | RQ | PQ|
| ------------------ |--- | --- |--- |
| **U-TAE** (ours)      | **81.5**|**53.2** |**43.8**|
| UConvLSTM+PaPs  | 80.2|   43.9   |  35.6|

### PASTIS - Semantic segmentation
The spatio-temporal encoder U-TAE yields a semantic segmentation score of 63.1 mIoU on PASTIS, achieving an improvement of approximately 5 points compared to the best existing methods that we re-implemented (Unet-3d, Unet+ConvLSTM and Feature Pyramid+Unet).


| Model name         | #Params| OA  |  mIoU |
| ------------------ |---- |---- | ---|
| **U-TAE**  (ours) |   **1.1M**|  **83.2%**    | **63.1%**|
| Unet-3d   | 1.6M|    81.3%    |  58.4%|
| Unet-ConvLSTM |1.5M  |     82.1%    |  57.8%|
| FPN-ConvLSTM  | 1.3M|    81.6%   |  57.1%|



## Training models from scratch

### Panoptic segmentation

To reproduce the main result for panoptic segmentation (with U-TAE+PaPs) run the following :

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```
Options are also provided in `train_panoptic.py` to reproduce the other results of Table 2:

```train
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_NoCNN --no_mask_conv
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --backbone uconvlstm
python train_panoptic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_shape24 --shape_size 24
```

Note: By default this script runs the 5 folds of the cross validation, which can be quite long (~12 hours per fold on a Tesla V100). 
Use the fold argument to execute one of the 5 folds only 
(e.g. for the 3rd fold : `python train_panoptic.py --fold 3 --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR`).

### Semantic segmentation

To reproduce results for semantic segmentation (with U-TAE) run the following :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR
```

And in order to obtain the results of the competing methods presented in Table 1 :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UNET3d --model unet3d
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_UConvLSTM --model uconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_FPN --model fpn
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_BUConvLSTM --model buconvlstm
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_COnvGRU --model convgru
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_ConvLSTM --model convlstm

```
Finally, to reproduce the ablation study presented in Table 1 :

```train
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_MeanAttention --agg_mode att_mean
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_SkipMeanConv --agg_mode mean
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_BatchNorm --encoder_norm batch
python train_semantic.py --dataset_folder PATH_TO_DATASET --res_dir OUT_DIR_SingleDate --mono_date "08-01-2019"

```

