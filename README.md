<div align="center">
<h1> XAMI: XMM-Newton optical Artefact Mapping for astronomical Instance segmentation </h1>
</div>

The code uses images from the XAMI dataset (available on [Github](https://github.com/ESA-Datalabs/XAMI-dataset) and [HuggingFace](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)). The images are astronomical observations from the Optical Monitor (XMM-OM) onboard the XMM-Newton X-ray mission. 

Information about the XMM-OM can be found here: 

- [The ESA website](https://www.cosmos.esa.int/web/xmm-newton/technical-details-om)
- [The article *The XMM-Newton optical/UV monitor telescope*](https://ui.adsabs.harvard.edu/abs/2001A%26A...365L..36M/abstract).

<p align="center">
  <img src="https://github.com/ESA-Datalabs/XAMI-model/blob/main/example_images/xami_model.png" alt="The XAMI model combining a detector and segmentor, while freezing the detector model previously trained on the XAMI dataset." width="40%">
</p>

<p align="center">
  <em>Figure 1: The XAMI model combining a detector and segmentor, while freezing the detector model previously trained on the XAMI dataset.</em>
</p>

## Cloning the repository

```bash
git clone https://github.com/ESA-Datalabs/XAMI-model.git
cd XAMI-model

# creating the environment
conda env create -f environment.yaml
conda activate xami_model_env

# Install the package in editable mode
pip install -e .
```

## Downloading the dataset and model checkpoints from HuggingFace

The dataset is splited into train and validation categories and contains annotated artefacts in COCO format for Instance Segmentation. We use multilabel Stratified K-fold (k=4) to balance class distributions across splits. We choose to work with a single dataset splits version (out of 4) but also provide means to work with all 4 versions.

The [Dataset-Structure.md](https://github.com/ESA-Datalabs/XAMI-dataset/blob/main/Datasets-Structure.md) offers more details about the dataset structure. We provide the following dataset formats: COCO format for Instance Segmentation (commonly used by [Detectron2](https://github.com/facebookresearch/detectron2) models) and YOLOv8-Seg format used by [ultralytics](https://github.com/ultralytics/ultralytics).

<!-- 1. **Downloading** the dataset archive from [HuggingFace](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset/blob/main/xami_dataset.zip).

```bash
DEST_DIR='.' # destination folder for the dataset (should usually be set to current directory)

huggingface-cli download iulia-elisa/XAMI-dataset xami_dataset.zip --repo-type dataset --local-dir "$DEST_DIR" && unzip "$DEST_DIR/xami_dataset.zip" -d "$DEST_DIR" && rm "$DEST_DIR/xami_dataset.zip"
``` -->


Check the [dataset_and_model.ipynb](https://github.com/ESA-Datalabs/XAMI-model/blob/main/dataset_and_model.ipynb) for downloading the dataset and model weights.

## Model Inference

After cloning the repository and setting up the environment, use the following code for model loading and inference:

```python
from xami_model.inference.xami_inference import InferXami

det_type = 'rtdetr' # 'rtdetr'  'yolov8'

detr_checkpoint = f'./xami_model/train/weights/{det_type}_sam_weights/{det_type}_detect_300e_best.pt'
sam_checkpoint = f'./xami_model/train/weights/{det_type}_sam_weights/{det_type}_sam.pth'

detr_sam_pipeline = InferXami(
    device='cuda:0',
    detr_checkpoint=detr_checkpoint,
    sam_checkpoint=sam_checkpoint,
    model_type='vit_t', # the SAM checkpoint and model_type (vit_h, vit_t, etc.) must be compatible
    use_detr_masks=True,
    detr_type=det_type)

masks = detr_sam_pipeline.run_predict('./example_images/S0893811101_M.png', show_masks=True)
```

For training the model, check the training [README.md](https://github.com/ESA-Datalabs/XAMI-model/blob/main/xami_model/train/README.md).

## Performance metrics 

<p align="center">
  <img src="https://github.com/ESA-Datalabs/XAMI-model/blob/main/example_images/ious_rtdetr.png" alt="Cumulative distribution of IoUs between predicted and true masks using RT-DETR as detector." width="70%">
</p>
<p align="center">
  <em>Figure 2: Cumulative distribution of IoUs between predicted and true masks using RT-DETR as detector.</em>
</p>

<div align="center">
<table style="width: 100%; border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th rowspan="2">Category</th>
      <th colspan="2">Precision</th>
      <th colspan="2">Recall</th>
    </tr>
    <tr>
      <th>YOLO-v8</th>
      <th>RT-DETR</th>
      <th>YOLO-v8</th>
      <th>RT-DETR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Overall</b></td>
      <td><b>84.3</b></td>
      <td><b>62.7</b></td>
      <td><b>72.1</b></td>
      <td><b>78.3</b></td>
    </tr>
    <tr>
      <td>Central-Ring</td>
      <td>89.3</td>
      <td>89.1</td>
      <td>94.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <td>Read-out-Streak</td>
      <td>71.1</td>
      <td>68.3</td>
      <td>73.3</td>
      <td>95.3</td>
    </tr>
    <tr>
      <td>Smoke-Ring</td>
      <td>80.6</td>
      <td>78.1</td>
      <td>85.6</td>
      <td>93.8</td>
    </tr>
    <tr>
      <td>Star-Loop</td>
      <td>80.5</td>
      <td>71.6</td>
      <td>74.1</td>
      <td>83.3</td>
    </tr>
    <tr>
      <td><i>Other</i></td>
      <td>100.0</td>
      <td>6.2</td>
      <td>33.3</td>
      <td>22.2</td>
    </tr>
  </tbody>
</table>
</div>

<p align="center">
  <em>Table 1: Metrics per object detector used.</em>
</p>

## © Licence 

This project is licensed under [MIT license](LICENSE).
