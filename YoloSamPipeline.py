import copy
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dataset import dataset_utils
from sam_predictor import predictor_utils
from ultralytics import YOLO, RTDETR
from astropy.io import fits
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
import torch.nn.functional as F
from segment_anything.utils.transforms import ResizeLongestSide
sys.path.append('/workspace/raid/OM_DeepLearning/MobileSAM-fine-tuning/')
from ft_mobile_sam import sam_model_registry, SamPredictor, build_efficientvit_l2_encoder

class YoloSam:
	def __init__(self, device, yolo_checkpoint, sam_checkpoint, model_type='vit_t', efficient_vit_enc=None, yolo_conf=0.2):
		self.device = device
		self.yolo_checkpoint = yolo_checkpoint
		self.sam_checkpoint = sam_checkpoint
		self.classes = {#0:('artefact', (252,252,214)),
						0:('central-ring', (1,252,214)), 
      					1:('other', (255,128,1)),
						2:('read-out-streak', (20, 77, 158)), 
      					3:('smoke-ring', (159,21,100)),
						4:('star-loop', (255, 188, 248)),
						5:('unknown', (0,0,0))	}
  
		self.use_yolo_masks = True # wether to use YOLO masks for faint sources
		self.yolo_conf = yolo_conf
		self.efficient_vit_enc = efficient_vit_enc	
  
		# Step 1: Object detection with YOLO
		if 'detr' in yolo_checkpoint:
			self.yolo_model = RTDETR(self.yolo_checkpoint) 
		else:
			self.yolo_model = YOLO(self.yolo_checkpoint) 

		# Step 2: Instance segmentation with SAM on detected objects
		self.mobile_sam_model, self.sam_predictor = self.load_sam_model(model_type)
		self.transform = ResizeLongestSide(self.mobile_sam_model.image_encoder.img_size)

	def load_sam_model(self, model_type="vit_t"):
		if self.efficient_vit_enc is not None:
			mobile_sam_model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint, efficient_vit_enc=self.efficient_vit_enc)
		else:
			mobile_sam_model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
   
		mobile_sam_model.to(self.device);
		mobile_sam_model.eval();
		predictor = SamPredictor(mobile_sam_model)
		return mobile_sam_model, predictor
		
	@torch.no_grad()
	def run_predict(self, image_path):

		start_time = datetime.now()
  
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		wt_mask, wt_image = dataset_utils.isolate_background(image, decomposition='db1', level=2, sigma=1) # wavelet decomposition for faint sources

		obj_results = self.yolo_model.predict(image_path, verbose=False, conf=self.yolo_conf) 
		self.sam_predictor.set_image(image)
		input_image = predictor_utils.transform_image(self.mobile_sam_model, self.transform, image, 'dummy_id', self.device)
		input_image = torch.as_tensor(input_image['image'], dtype=torch.float, device=self.device) # (B, C, 1024, 1024)

		# sets a specific mean for each image
		image_T = np.transpose(image, (2, 1, 0))
		mean_ = np.mean(image_T[image_T>0])
		std_ = np.std(image_T[image_T>0]) 
		pixel_mean = torch.as_tensor([mean_, mean_, mean_], dtype=torch.float, device=self.device)
		pixel_std = torch.as_tensor([std_, std_, std_], dtype=torch.float, device=self.device)

		self.mobile_sam_model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
		self.mobile_sam_model.register_buffer("pixel_std", torch.Tensor(pixel_std).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
			
        # IMAGE ENCODER
		# image_embedding = self.mobile_sam_model.image_encoder(input_image) (1, 3, 1024, 1024)
  
		if len(obj_results[0]) == 0:
			print("No objects detected. Check model configuration or input image.")
			return None
		return 1, 2

		input_boxes1 = obj_results[0].boxes.xyxy
		predicted_classes = obj_results[0].boxes.cls
		colours = [self.classes[i.item()][1] for i in predicted_classes]
		expand_by = 0.0
		enlarged_bbox = input_boxes1.clone() 
		enlarged_bbox[:, :2] -= expand_by  
		enlarged_bbox[:, 2:] += expand_by  
		input_boxes1 = enlarged_bbox
		input_boxes = input_boxes1.cpu().numpy()
		input_boxes = self.sam_predictor.transform.apply_boxes(input_boxes, self.sam_predictor.original_size)
		input_boxes = torch.from_numpy(input_boxes).to(self.device)
		sam_mask = []
  
		if self.use_yolo_masks:
			yolo_masks = []
			non_resized_masks = obj_results[0].masks.data.cpu().numpy()
			for i in range(len(non_resized_masks)):
				yolo_masks.append(cv2.resize(non_resized_masks[i], image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)) 

		image_embedding=self.sam_predictor.features # torch.Size([1, 256, 64, 64])
		prompt_embedding=self.mobile_sam_model.prompt_encoder.get_dense_pe() # torch.Size([1, 256, 64, 64]) 
  
		sparse_embeddings, dense_embeddings = self.mobile_sam_model.prompt_encoder( # torch.Size([N, 2, 256]),  torch.Size([N, 256, 64, 64])
				points=None,
				boxes=input_boxes,
				masks=None,) 
  
		low_res_masks, iou_predictions = self.mobile_sam_model.mask_decoder(
								image_embeddings=image_embedding,
								image_pe=prompt_embedding,
								sparse_prompt_embeddings=sparse_embeddings,
								dense_prompt_embeddings=dense_embeddings,
								multimask_output=True,
							)
		max_low_res_masks = torch.zeros((low_res_masks.shape[0], 1, 256, 256))
		max_ious = torch.zeros((iou_predictions.shape[0], 1))

		# Take all low_res_mask correspnding to the index of max_iou
		for i in range(low_res_masks.shape[0]):
			max_iou_index = torch.argmax(iou_predictions[i])
			max_low_res_masks[i] = low_res_masks[i][max_iou_index].unsqueeze(0)
			max_ious[i] = iou_predictions[i][max_iou_index]

		low_res_masks = max_low_res_masks
		iou_predictions = max_ious
		print('Number of object detected:', len(iou_predictions))
		print('Iou predictions:', iou_predictions.flatten())
		print("Unique classes detected:", len(predicted_classes.unique()))
		for predicted_class in predicted_classes.unique():
			rgb = self.classes[predicted_class.item()][1]
			label = self.classes[predicted_class.item()][0]
			escape_code = f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m \x1b[0m'
			print(escape_code+escape_code, label, end='\n\n')
		print("self.sam_predictor.input_size", self.sam_predictor.input_size)
   
		low_res_masks=self.sam_predictor.model.postprocess_masks(
      	   low_res_masks, 
           self.sam_predictor.input_size, 
           self.sam_predictor.original_size).to(self.device)
  
		if self.use_yolo_masks:
			# take faint masks from YOLO
			for i in range(len(low_res_masks)):
				thresholded_mask = (low_res_masks[i]>0.5).detach().cpu().numpy()
				# plt.imshow(yolo_masks[i], cmap='gray')	
				# plt.show()	
				# plt.close()
				wt_on_mask = wt_mask[:, :, 0] * thresholded_mask/ np.sum(thresholded_mask[0])
				wt_count = np.sum(wt_on_mask)
				if (wt_count > 0.6 and predicted_classes[i] in [1.0, 2.0, 4.0]): #or predicted_classes[i] in [1.0, 2.0]: 
					low_res_masks[i] = torch.as_tensor(yolo_masks[i]).to(self.device).unsqueeze(0)

		# Apply Gaussian filter on logits
		kernel_size, sigma = 5, 2
		gaussian_kernel = predictor_utils.create_gaussian_kernel(kernel_size, sigma).to(self.device)
		pred_masks = torch.nn.functional.conv2d(low_res_masks, gaussian_kernel, padding=kernel_size//2)
		threshold_masks = torch.sigmoid(10 * (pred_masks - self.mobile_sam_model.mask_threshold)) # sigmoid with steepness
		sam_mask_pre = (threshold_masks > 0.5)*1.0
		print("Inference time:", datetime.now()-start_time)
		sam_mask.append(sam_mask_pre.squeeze(1))
		sam_masks_numpy = sam_mask[0].detach().cpu().numpy()
  
		if len(sam_masks_numpy) == 0:
			print("No masks detected. Check model configuration or input image.")
			return None

		fig, axes = plt.subplots(1, 3, figsize=(20, 8)) 
		image_copy = image.copy()

		# # Plot 4: SAM Masks
		# axes[0].imshow(image)
		# axes[0].set_title('Image')
		# # plot yolobounding boxes on the image with class names and Iou predictions	
  
		# for i in range (len(input_boxes1)):
		# 	dataset_utils.visualize_titles(image_copy, input_boxes1[i], self.classes[predicted_classes[i].item()][0], font_thickness = 1, font_scale=0.35)
		# axes[1].imshow(image_copy)
		# for i in range (len(input_boxes1)):
		# 	dataset_utils.show_box(input_boxes1[i].detach().cpu().numpy(), axes[1])
		# axes[2].imshow(image)
		# dataset_utils.show_masks(sam_masks_numpy, axes[2], random_color=False, colours=colours)
		# axes[2].set_title('Yolo-SAM predicted masks')
		# plt.tight_layout() 
		# # plt.savefig(f'./{image_path.split("/")[-1].replace(".png", "_predicted.png")}')
		# plt.show()
		return sam_mask_pre, obj_results # obj_results for further inference 