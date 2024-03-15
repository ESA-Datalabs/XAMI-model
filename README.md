---
language: en
license: mit
datasets:
- iulia-elisa/AstroArtefactToolkit_XMMoptical
model-index:
- name: xmm_om_model
  results:
  - task:
      name: Instance Segmentation
      type: instance-segmentation
    dataset:
      name: AstroArtefactToolkit_XMMoptical 
      type: AstroArtefactToolkit_XMMoptical
    metrics:
       - name: Accuracy
         type: accuracy
         value: model_accuracy
---
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

```shell
git clone https://huggingface.co/iulia-elisa/xmm_om_models
cd xmm_om_model
pip install -r requirements.txt
```

## Running the Model Pipeline

After setting up your environment, use the following Python code to dynamically load the models and custom processing steps.

```python
import sys
from YoloSamPipeline import YoloSam

yolo_path = 'path/to/yolo/checkpoint' 
mobile_sam_path = 'path/to/mobile_sam/checkpoint' 

yolo_sam_pipe = YoloSam('cuda:0', yolo_path, mobile_sam_path )
yolo_sam_pipe.run_predict('path/to/image')
```

## Licence 

license: cc-by-4.0

