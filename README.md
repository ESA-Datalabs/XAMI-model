# XAMI (**X**MM-Newton optical **A**rtefact **M**apping for astronomical **I**nstance segmentation)

The code uses images from the XAMI dataset (available on [Github](https://github.com/ESA-Datalabs/XAMI-dataset) and [HuggingFace🤗](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)). The images represent observations from the XMM-Newton's Opical Monitor (XMM-OM). Information about the XMM-OM can be found here: 

- XMM-OM User's Handbook: https://www.mssl.ucl.ac.uk/www_xmm/ukos/onlines/uhb/XMM_UHB/node1.html.
- Technical details: https://www.cosmos.esa.int/web/xmm-newton/technical-details-om.
- The article https://ui.adsabs.harvard.edu/abs/2001A%26A...365L..36M/abstract.

## Cloning the repository

```bash
git clone https://github.com/ESA-Datalabs/XAMI.git
cd XAMI

# creating the environment
conda env create -f environment.yml
conda activate xami_env
```

## Model Inference

After cloning the repository and setting up your environment, use the following Python code for model loading and inference (see [yolo_sam_inference.ipynb](https://github.com/ESA-Datalabs/XAMI/blob/main/yolo_sam_inference.ipynb)).

```python
import sys
from inference.YoloSamPipeline import YoloSam

yolo_checkpoint = './train/weights/yolo_weights/best.pt'
sam_checkpoint = './train/weights/sam_weights/sam_0_best.pth'
device_id = 0

# the SAM model checkpoint and model_type (vit_h, vit_t, etc.) must be compatible
yolo_sam_pipeline = YoloSam(
    device=f'cuda:{device_id}', 
    yolo_checkpoint=yolo_checkpoint, 
    sam_checkpoint=sam_checkpoint, 
    model_type='vit_t')

# prediction example
masks = yolo_sam_pipeline.run_predict(
    './example_images/S0743200101_V.jpg', 
    yolo_conf=0.2, 
    show_masks=True)
```

## Training the model

1. **Downloading** the dataset archive from [HuggingFace](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset/blob/main/xami_dataset.zip).

```bash
DEST_DIR='.' # destination folder for the dataset (should usually be set to current directory)

huggingface-cli download iulia-elisa/XAMI-dataset xami_dataset.zip --repo-type dataset --local-dir "$DEST_DIR" && unzip "$DEST_DIR/xami_dataset.zip" -d "$DEST_DIR" && rm "$DEST_DIR/xami_dataset.zip"
```

2. **Training**.

Check the training [README.md](https://github.com/ESA-Datalabs/XAMI/blob/main/train/README.md).

## Licence 

This project is licensed under [MIT license](LICENSE).

