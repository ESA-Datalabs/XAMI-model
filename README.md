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

## Running the Model Pipeline

After cloning the repository and setting up your environment, use the following Python code for model loading and inference.

```python
import sys
from inference.YoloSamPipeline import YoloSam

yolo_checkpoint = './train/yolov8-segm-0/yolov8n-seg/weights/best.pt'
sam_checkpoint = './output_sam/ft_mobile_sam_final_2024-05-05 18:38:00.526813.pth'
device_id = 3

# the checkpoint and model_type (vit_h, vit_t, etc.) must be compatible
yolo_sam_pipeline = YoloSam(
    device=f'cuda:{device_id}', 
    yolo_checkpoint=yolo_checkpoint, 
    sam_checkpoint=sam_checkpoint, 
    model_type='vit_t')

# predict
masks = yolo_sam_pipeline.run_predict('./path/to/image', yolo_conf=0.2, show_masks=True)
```

## Licence 

This project is licensed under [MIT license](LICENSE).

