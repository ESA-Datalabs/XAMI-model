import typer
from mmengine.hub import get_config
from mmengine.runner import Runner
from importlib import import_module
module = import_module(f'mmseg.utils')
module.register_all_modules(False)
import os
import mmcv

# Check cuda and mmcv-full visibility
import torch
print(torch.cuda.is_available()) 
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))
# import logging
# logging.basicConfig(level=logging.ERROR)  # Display only errors
# logger = logging.getLogger('mmcv')  # Get MMCV's logger
# logger.setLevel(logging.WARNING)  # Adjust logging level

# logger = logging.getLogger('mmdet')  # Get MMDetection's logger
# # logger.setLevel(logging.WARNING)  # Adjust logging level
# import mmcv
# mmcv.utils.get_logger('mmdet')
# logger.setLevel('WARNING')

# # Check if MMCV is compiled with CUDA support
# cuda_available = mmcv.ops.is_cuda_available()
# print(f"CUDA Available: {cuda_available}")

# Optionally, check for a specific op to ensure it's compiled; for example, RoIAlign
try:
    from mmcv.ops import RoIAlign
    print("RoIAlign CUDA op is available.")
except ImportError as e:
    print(f"Error importing RoIAlign: {e}")


# This is just needed to fix the num_classes in a model 
# You'd really hope that there's a better approach 
# But I'm not sure, and have yet to see one. 
# After a good bit of looking.
def replace_nested_value(dictionary, key_to_replace, new_value):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            if key == key_to_replace:
                dictionary[key] = new_value
            elif isinstance(value, dict):
                replace_nested_value(value, key_to_replace, new_value)
            elif isinstance(value, list):
                for item in value:
                    replace_nested_value(item, key_to_replace, new_value)

def set_num_frozen_stages(model: dict, frozen_stages: int): 
    model['backbone']['frozen_stages'] = frozen_stages
    return model

def main(
    data_root: str = './xmm_om_images_512_SG_SR_CR_only-10-COCO',
    data_prefix: str = 'images', # dict(img="data/")
    training_path: str = 'train/_annotations.coco.json',
    validation_path: str = 'valid/_annotations.coco.json',
    test_path: str = 'test/_annotations.coco.json',
    model_architecture_family: str = 'swin',
    training_config: str = "mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco.py",
    num_classes: int = 4,
    output_directory: str = 'work_dir',
    frozen_stages: int = 3,
):


	# optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)

    cfg = get_config(f'mmdet::{model_architecture_family}/{training_config}')

    # Set the number of frozen stages in the model
    cfg.model = set_num_frozen_stages(cfg.model, frozen_stages)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    cfg.work_dir = output_directory

    train_dataset=dict(
        type="CocoDataset",
        data_root=data_root, 
        ann_file=training_path,
        data_prefix=dict(img="train"),
    )
    val_dataset=dict(
        type="CocoDataset",
        data_root=data_root, 
        ann_file=validation_path,
        data_prefix=dict(img="valid"),
    )

    test_dataset=dict(
        type="CocoDataset",
        data_root=data_root, 
        ann_file=test_path,
        data_prefix=dict(img="test"),
		test_mode=True,
		
    )


    train_pipeline = [
	    dict(type='LoadImageFromFile'),
	    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
		dict(type='PackDetInputs')]

    test_pipeline = [
	    dict(type='LoadImageFromFile'),
	    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
	    dict(
	        type='PackDetInputs',
	        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
	                   'scale_factor'))
	]

    cfg.train_dataloader.dataset = train_dataset
    cfg.train_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+training_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}

    cfg.train_dataloader.dataset.pipeline = train_pipeline
 #    log_config = dict(
	#     interval=50,  # Log every 50 iterations
	#     hooks=[
	#         dict(type='TextLoggerHook'),
	#         # If you have TensorBoard or Wandb logger, configure them here as well
	#     ],
	#     level='ERROR',  # Adjust logging level
	# )

    cfg.val_dataloader.dataset = val_dataset 
    cfg.val_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+validation_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}
    cfg.val_dataloader.dataset.pipeline = test_pipeline

    cfg.test_dataloader.dataset = test_dataset #.data_root = data_root
    cfg.test_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+test_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    log_config = dict(
	    interval=1000,
	    hooks=[
        # dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
    cfg.log_config = log_config
    replace_nested_value(cfg.model, 'num_classes', num_classes)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
   typer.run(main)