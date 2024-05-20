# XAMI (**X**MM-Newton Optical **A**rtefact **M**apping for Astronomical **I**nstance Segmentation)

The code uses images from the XAMI dataset (available on [Github](https://github.com/ESA-Datalabs/XAMI-dataset) and [HuggingFace🤗](https://huggingface.co/datasets/iulia-elisa/XAMI-dataset)). The images are telescope observation from the XMM-Newton's Opical Monitor (XMM-OM). Information about the XMM-OM can be found here: 

- XMM-OM User's Handbook: https://www.mssl.ucl.ac.uk/www_xmm/ukos/onlines/uhb/XMM_UHB/node1.html.
- Technical details: https://www.cosmos.esa.int/web/xmm-newton/technical-details-om.
- The article https://ui.adsabs.harvard.edu/abs/2001A%26A...365L..36M/abstract.

## Cloning the repository

```
git clone https://github.com/ESA-Datalabs/XAMI.git
cd XAMI
```

## Creating the environment

```bash
conda env create -f environment.yml
conda activate xami_env
```

## Running the Model Pipeline

After cloning the repository and setting up your environment, use the following Python code for model loading and inference.

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
masks = yolo_sam_pipeline.run_predict('./example_images/S0743200101_V.jpg', yolo_conf=0.2, show_masks=True)
```

## Training the model

Check the training [README.md](https://github.com/ESA-Datalabs/XAMI/blob/main/train/README.md).

## Licence 

This project is licensed under [MIT license](LICENSE).

