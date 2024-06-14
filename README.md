<div align="center">
<h1> XAMI: XMM-Newton optical Artefact Mapping for astronomical Instance segmentation </h1>
</div>

## 💫 Introduction
The code uses images from the XAMI dataset (available on [Github](https://github.com/ESA-Datalabs/XAMI-dataset) and [HuggingFace🤗](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)). The images represent observations from the XMM-Newton's Opical Monitor (XMM-OM). Information about the XMM-OM can be found here: 

- XMM-OM User's Handbook: https://www.mssl.ucl.ac.uk/www_xmm/ukos/onlines/uhb/XMM_UHB/node1.html.
- Technical details: https://www.cosmos.esa.int/web/xmm-newton/technical-details-om.
- The article https://ui.adsabs.harvard.edu/abs/2001A%26A...365L..36M/abstract.

## 📂 Cloning the repository

```bash
git clone https://github.com/ESA-Datalabs/XAMI-model.git
cd XAMI-model

# creating the environment
conda env create -f environment.yaml
conda activate xami_model_env

# Install the package in editable mode
pip install -e .
```

## 📊 Downloading the dataset and model checkpoints from HuggingFace🤗

Check [dataset_and_model.ipynb](https://github.com/ESA-Datalabs/XAMI-model/blob/main/dataset_and_model.ipynb) for downloading the dataset and model weights. 

The dataset is splited into train and validation categories and contains annotated artefacts in COCO format for Instance Segmentation. We use multilabel Stratified K-fold (k=4) to balance class distributions across splits. We choose to work with a single dataset splits version (out of 4) but also provide means to work with all 4 versions.

To better understand our dataset structure, please check the [Dataset-Structure.md](https://github.com/ESA-Datalabs/XAMI-dataset/blob/main/Datasets-Structure.md) for more details. We provide the following dataset formats: COCO format for Instance Segmentation (commonly used by [Detectron2](https://github.com/facebookresearch/detectron2) models) and YOLOv8-Seg format used by [ultralytics](https://github.com/ultralytics/ultralytics).

<!-- 1. **Downloading** the dataset archive from [HuggingFace](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset/blob/main/xami_dataset.zip).

```bash
DEST_DIR='.' # destination folder for the dataset (should usually be set to current directory)

huggingface-cli download iulia-elisa/XAMI-dataset xami_dataset.zip --repo-type dataset --local-dir "$DEST_DIR" && unzip "$DEST_DIR/xami_dataset.zip" -d "$DEST_DIR" && rm "$DEST_DIR/xami_dataset.zip"
``` -->

## 💡 Model Inference

After cloning the repository and setting up the environment, use the following Python code for model loading and inference:

```python
from xami_model.inference.xami_inference import InferXami

detr_checkpoint = './xami_model/train/weights/yolo_weights/last.pt'
sam_checkpoint = './xami_model/train/weights/sam_weights/sam_0_best.pt'

# the SAM checkpoint and model_type (vit_h, vit_t, etc.) must be compatible
detr_sam_pipeline = InferXami(
    device='cuda:0',
    detr_checkpoint=detr_checkpoint,
    sam_checkpoint=sam_checkpoint,
    model_type='vit_t',
    use_detr_masks=False)

masks = detr_sam_pipeline.run_predict('./example_images/S0893811101_M.png', show_masks=True)
```

## 🚀 Training the model

Check the training [README.md](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/README.md).

## © Licence 

This project is licensed under [MIT license](LICENSE).

