# XMM_OM_AI Deep Learning project

The code uses images from the XMM-Newton's Opical Monitor. The image windows are stacked (2048x2048) and rebinned to 512x512.
Information about the configuration of the OM (e.g.: sub-windows stacking) can be found here: 


> https://www.mssl.ucl.ac.uk/www_xmm/ukos/onlines/uhb/XMM_UHB/node62.html
>
> https://www.cosmos.esa.int/web/xmm-newton/technical-details-om
>
> https://heasarc.gsfc.nasa.gov/docs/xmm/uhb/om.html
> 

## Clone the repo:

```
git clone https://github.com/ESA-Datalabs/XAMI.git
cd XAMI
```

## Running the Model Pipeline

After setting up your environment, use the following Python code for model loading and inference.

```python
import sys
from YoloSamPipeline import YoloSam

yolo_path = 'path/to/yolo/checkpoint' 
mobile_sam_path = 'path/to/mobile_sam/checkpoint' 

# load the models
yolo_sam_pipeline = YoloSam(
    device='cuda:0', 
    yolo_checkpoint=yolo_path, 
    sam_checkpoint='./output_sam/ft_mobile_sam_final_2024-04-27 00:02:11.627528_last.pth', # the checkpoint and model_type (vit_h, vit_t, etc.) must be compatible
    model_type='vit_t',
    efficient_vit_enc=None,
    yolo_conf=0.3)
# predict
yolo_sam_pipe.run_predict('path/to/image')
```

## Licence 

This project is licensed under [MIT license](LICENSE).

