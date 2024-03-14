import sys
import torch
import cv2
import matplotlib.pyplot as plt
sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-master/')
from mobile_sam import sam_model_registry, SamPredictor
from .data_preprocess import preprocess_utils
from .dataset import dataset_utils
from .class_agnostic_sam_predictor import predictor_utils

from ultralytics import YOLO

class YoloSamPipeline:
	def __init__(self, device, yolo_checkpoint, sam_checkpoint):
		self.device = device
		self.yolo_checkpoint = yolo_checkpoint
		self.sam_checkpoint = sam_checkpoint

		# Step 1: Object detection with YOLO
		self.yolo_model = YOLO(self.yolo_checkpoint) 

		# Step 2: Instance segmentation with SAM on detected objects
		self.mobile_sam_model, self.sam_predictor = self.load_sam_model()

	def load_sam_model(self, model_type="vit_t"):
		mobile_sam_model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
		mobile_sam_model.to(self.device);
		mobile_sam_model.eval();
		predictor = SamPredictor(mobile_sam_model)
		return mobile_sam_model, predictor
		
	@torch.no_grad()
	def run_predict(self, image_path):

		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		obj_results = self.yolo_model.predict(image_path, verbose=False, conf=0.2) 
		self.sam_predictor.set_image(image)
		# sets a specific mean for each image
		# image_T = np.transpose(image, (2, 1, 0))
		# mean_ = np.mean(image_T[image_T>0])
		# std_ = np.std(image_T[image_T>0]) 
		# pixel_mean = torch.as_tensor([mean_, mean_, mean_], dtype=torch.float, device=device)
		# pixel_std = torch.as_tensor([std_, std_, std_], dtype=torch.float, device=device)

		# mobile_sam_model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
		# mobile_sam_model.register_buffer("pixel_std", torch.Tensor(pixel_std).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
			
		if len(obj_results[0]) == 0:
			# No objects detected
			return None

		input_boxes1 = obj_results[0].boxes.xyxy
		expand_by = 1.5
		enlarged_bbox = input_boxes1.clone() 
		enlarged_bbox[:, :2] -= expand_by  
		enlarged_bbox[:, 2:] += expand_by  
		input_boxes1 = enlarged_bbox
		input_boxes = input_boxes1.cpu().numpy()
		input_boxes = self.sam_predictor.transform.apply_boxes(input_boxes, self.sam_predictor.original_size)
		input_boxes = torch.from_numpy(input_boxes).to(self.device)
		sam_mask = []

		image_embedding=self.sam_predictor.features
		prompt_embedding=self.mobile_sam_model.prompt_encoder.get_dense_pe()
		non_resized_masks = obj_results[0].masks.data.cpu().numpy()

		sparse_embeddings, dense_embeddings = self.mobile_sam_model.prompt_encoder(
				points=None,
				boxes=input_boxes,
				masks=None,)

		low_res_masks, _ = self.mobile_sam_model.mask_decoder(
								image_embeddings=image_embedding,
								image_pe=prompt_embedding,
								sparse_prompt_embeddings=sparse_embeddings,
								dense_prompt_embeddings=dense_embeddings,
								multimask_output=False,
							)
  
		low_res_masks=self.sam_predictor.model.postprocess_masks(low_res_masks, self.sam_predictor.input_size, self.sam_predictor.original_size)
		threshold_masks = torch.sigmoid(low_res_masks - self.mobile_sam_model.mask_threshold) 
		sam_mask_pre = (threshold_masks > 0.5)*1.0
		sam_mask.append(sam_mask_pre.squeeze(1))
  
		fig, axes = plt.subplots(1, 2, figsize=(18, 6)) 

		# Plot 4: SAM Masks
		sam_masks_numpy = sam_mask[0].detach().cpu().numpy()
		axes[3].imshow(image)
		predictor_utils.show_masks(sam_masks_numpy, axes[3], random_color=True)
		axes[3].set_title('MobileSAM predicted masks')
		plt.tight_layout() 
		# plt.savefig(f'./plots/combined_plots.png')
		plt.show()