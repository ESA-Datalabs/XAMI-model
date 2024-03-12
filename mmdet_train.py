import typer
from mmengine.hub import get_config
from mmengine.runner import Runner
from importlib import import_module
module = import_module(f'mmseg.utils')
module.register_all_modules(False)
import os
import mmcv
# from mmcv.runner.hooks import HOOKS, Hook

# Check torch cuda visibility

import torch
print(torch.cuda.is_available()) 
print(torch.cuda.device_count())  
print(torch.cuda.get_device_name(0))

# Check MMCV CUDA Compiler visibility
try:
    from mmcv.ops import RoIAlign
    print("RoIAlign CUDA op is available.")
except ImportError as e:
    print(f"Error importing RoIAlign: {e}")

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


# @HOOKS.register_module()
# class WandBLoggerHook(Hook):
#     def after_train_epoch(self, runner):
#         metrics = runner.log_buffer.output
#         wandb.log(metrics, step=runner.epoch)

def set_num_frozen_stages(model: dict, frozen_stages: int): 
    model['backbone']['frozen_stages'] = frozen_stages
    return model

def main(
    data_root: str = './xmm_om_images_512_SG_SR_CR_only-10-COCO',
    data_prefix: str = 'images',
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

    test_pipeline = [ # check this
	    dict(type='LoadImageFromFile'),
	    dict(
	        type='PackDetInputs',
	        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
	                   'scale_factor'))
	]

    cfg.train_dataloader.dataset = train_dataset
    cfg.train_dataloader.dataset.pipeline = train_pipeline
    cfg.train_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+training_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}

    cfg.val_dataloader.dataset = val_dataset 
    cfg.val_dataloader.dataset.pipeline = test_pipeline
    cfg.val_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+validation_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}
	
    cfg.test_dataloader.dataset = test_dataset
    cfg.test_dataloader.dataset.pipeline = test_pipeline
    cfg.test_evaluator = {
		'type': 'CocoMetric',
		'ann_file': data_root+'/'+test_path,
		'metric': ['bbox', 'segm'],
		'format_only': False,
		'backend_args': None ,
	}
	
    log_config = dict(
	    interval=1000,
	    hooks=[
        # dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
	
    cfg.log_config = log_config

	# track metrics with Wandb
    # cfg.custom_hooks = [dict(type='WandBLoggerHook')]


    replace_nested_value(cfg.model, 'num_classes', num_classes)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
   typer.run(main)