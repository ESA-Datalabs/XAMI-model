import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO, RTDETR
from segment_anything.utils.transforms import ResizeLongestSide
import tqdm
import os
from ..dataset import dataset_utils
from ..model_predictor import predictor_utils
from ..mobile_sam.mobile_sam import sam_model_registry, SamPredictor 

class InferXami:
  def __init__(self, device, detr_checkpoint, sam_checkpoint, model_type='vit_t', use_detr_masks=False, detr_type='yolo'):
    
    assert detr_type in ['yolo', 'yolov8', 'rt_detr', 'rtdetr'], "Invalid DETR type. Please choose either 'yolo', 'yolov8', 'rt_detr' or 'rtdetr'."
    print("Initializing the model...")

    self.device = device
    self.detr_checkpoint = detr_checkpoint
    self.sam_checkpoint = sam_checkpoint
    self.classes = {0:('central-ring', (1,252,214)), 
                    1:('other', (255,128,1)),
                    2:('read-out-streak', (20, 77, 158)), 
                    3:('smoke-ring', (159,21,100)),
                    4:('star-loop', (255, 188, 248))}

    self.use_detr_masks = use_detr_masks # whether to use YOLO masks for faint sources
    
    # Step 1: Object detection
    if detr_type.startswith('yolo'):
      self.detector = YOLO(self.detr_checkpoint)
      if self.use_detr_masks==True:
        print("Using YOLO masks for faint (1sigma) sources.")
    elif detr_type == 'rt_detr' or detr_type == 'rtdetr':
      self.detector = RTDETR(self.detr_checkpoint)
      if self.use_detr_masks==True:
        print("RT-DETR doesn't currently support masks generation. We will set use_detr_masks=False.")
        self.use_detr_masks=False

    self.detector.to(self.device)
    # Step 2: Instance segmentation with SAM on detected objects
    self.mobile_sam_model, self.model_predictor = self.load_sam_model(model_type)
    self.transform = ResizeLongestSide(self.mobile_sam_model.image_encoder.img_size)
    
    # Warmup is beneficial for completing system-level optimizations
    self.model_warmup()

  def load_sam_model(self, model_type="vit_t"):
    
    mobile_sam_model = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
    mobile_sam_model.to(self.device);
    mobile_sam_model.eval();
    predictor = SamPredictor(mobile_sam_model)
    
    return mobile_sam_model, predictor
  
  def model_warmup(self, number_of_runs=3):
    """
    Warm up the YOLO and SAM models by running them with random input multiple times.

    Args:
      number_of_runs (int): The number of times to run the models with random input. Default is 3.

    Returns:
      None
    """
    # Warm up YOLO model
    for _ in tqdm.tqdm(range(number_of_runs), desc="Warming up YOLO model", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
      _ = self.detector.predict('./xami_model/inference/warmup_image.png', verbose=False, conf=0.2) 

    # Warm up SAM model
    for _ in tqdm.tqdm(range(number_of_runs), desc="Warming up SAM model", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
      _ = self.run_sam_model(
        torch.randn(1, 3, 1024, 1024).to(self.device),
        torch.tensor([[0, 0, 512, 512]]).to(self.device)
      )
      
  @torch.no_grad()
  def run_predict(self, image_path, yolo_conf=0.2, show_masks=False):

    start_time_all = time.time()
    image = cv2.imread(image_path)
    obj_results = self.detector.predict(image_path, verbose=False, conf=yolo_conf) 

    # set a specific mean for each image
    input_image = predictor_utils.set_mean_and_transform(image, self.mobile_sam_model, self.transform, self.device)
             
    if len(obj_results[0]) == 0: # type: ignore
      print("No objects detected. Check model configuration or input image.")
      return None, None, (time.time()-start_time_all)*1000, 1

    predicted_classes = obj_results[0].boxes.cls
    colours = [self.classes[i.item()][1] for i in predicted_classes] # type: ignore
    boxes_numpy = obj_results[0].boxes.xyxy.cpu().numpy()
    input_boxes = self.model_predictor.transform.apply_boxes(boxes_numpy, image.shape[:-1])
    input_boxes = torch.from_numpy(input_boxes).to(self.device)
    sam_mask = []
    if self.use_detr_masks:
      yolo_masks = []
      non_resized_masks = obj_results[0].masks.data.cpu().numpy()
      for i in range(len(non_resized_masks)):
        yolo_masks.append(cv2.resize(non_resized_masks[i], image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)) 
    
    low_res_masks, iou_predictions = self.run_sam_model(input_image, input_boxes)

    print('Number of object detected:', len(iou_predictions))
    for predicted_class in predicted_classes.unique():
      rgb = self.classes[predicted_class.item()][1]
      escape_code = f'\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m \x1b[0m'
      print(escape_code+escape_code, self.classes[predicted_class.item()][0], end='\n')
        
    low_res_masks=self.model_predictor.model.postprocess_masks(low_res_masks, (1024, 1024), image.shape[:-1]).to(self.device)
    
    if self.use_detr_masks:
      low_res_masks = predictor_utils.process_faint_masks(
        image, 
        low_res_masks, 
        yolo_masks, 
        predicted_classes, 
        self.device,
        wt_threshold=0.6, 
        wt_classes=[1.0, 2.0, 4.0])

    # Apply Gaussian filter on logits
    kernel_size, sigma = 5, 2
    gaussian_kernel = predictor_utils.create_gaussian_kernel(kernel_size, sigma).to(self.device)
    pred_masks = torch.nn.functional.conv2d(low_res_masks, gaussian_kernel, padding=kernel_size//2)
    threshold_masks = torch.sigmoid(10 * (pred_masks - self.mobile_sam_model.mask_threshold)) # sigmoid with steepness
    sam_mask_pre = (threshold_masks > 0.5)*1.0
    inference_time = (time.time()-start_time_all)*1000
    # print(f"Total Inference time:: {inference_time:.2f} ms")
    sam_mask.append(sam_mask_pre.squeeze(1))
    sam_masks_numpy = sam_mask[0].detach().cpu().numpy()

    if len(sam_masks_numpy) == 0:
      print("No masks detected. Check model configuration or input image.")
      return None

    if show_masks:
      fig, axes = plt.subplots(1, 3, figsize=(20, 8)) 
      image_copy = image.copy()

      # Plot 4: SAM Masks
      axes[0].imshow(image)
      axes[0].set_title('Image')
      # plot yolobounding boxes on the image with class names and Iou predictions	

      for i in range (len(boxes_numpy)):
        dataset_utils.visualize_titles(
          image_copy, 
          boxes_numpy[i], 
          self.classes[predicted_classes[i].item()][0],  # type: ignore
          font_thickness=1, 
          font_scale=1.0)
        
      axes[1].imshow(image_copy)
      for i in range (len(boxes_numpy)):
        dataset_utils.show_box(boxes_numpy[i], axes[1])
      axes[2].imshow(image)
      dataset_utils.show_masks(sam_masks_numpy, axes[2], random_color=False, colours=colours)
      axes[2].set_title('Yolo-SAM predicted masks')
      plt.tight_layout() 
      plt.savefig(f'./{image_path.split("/")[-1].replace(".png", "_predicted.png")}')
      plt.show()
      
    return sam_mask_pre, obj_results, inference_time, 0 # obj_results for further inference 
  
  def run_sam_model(
    self, 
    input_image, 
    input_boxes,
    ):
    
    image_embedding = self.mobile_sam_model.image_encoder(input_image) # [1, 256, 64, 64]
           
    sparse_embeddings, dense_embeddings = self.mobile_sam_model.prompt_encoder(
        points=None,
        boxes=input_boxes,
        masks=None,) 
    
    low_res_masks, iou_predictions = self.mobile_sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.mobile_sam_model.prompt_encoder.get_dense_pe(),
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
    
    return low_res_masks, iou_predictions
  
  @torch.no_grad()    
  def process_source_extractor_prompts(self, image_path, boxes_numpy, show_masks = False):
      
      image = cv2.imread(image_path)
      start_time_all = time.time()

      # set a specific mean for each image
      input_image = predictor_utils.set_mean_and_transform(image, self.mobile_sam_model, self.transform, self.device)
              
      if len(boxes_numpy) == 0: # type: ignore
        print("No objects detected. Check model configuration or input image.")
        return None
      
      input_boxes = self.model_predictor.transform.apply_boxes(boxes_numpy, image.shape[:-1])
      input_boxes = torch.from_numpy(input_boxes).to(self.device)
      sam_mask = []
     
      low_res_masks, iou_predictions = self.run_sam_model(input_image, input_boxes)
      low_res_masks=self.model_predictor.model.postprocess_masks(low_res_masks, (1024, 1024), image.shape[:-1]).to(self.device)
      
      # Apply Gaussian filter on logits
      kernel_size, sigma = 5, 2
      gaussian_kernel = predictor_utils.create_gaussian_kernel(kernel_size, sigma).to(self.device)
      pred_masks = torch.nn.functional.conv2d(low_res_masks, gaussian_kernel, padding=kernel_size//2)
      threshold_masks = torch.sigmoid(10 * (pred_masks - self.mobile_sam_model.mask_threshold)) # sigmoid with steepness
      sam_mask_pre = (threshold_masks > 0.5)*1.0
      inference_time = (time.time()-start_time_all)*1000
      sam_mask.append(sam_mask_pre.squeeze(1))
      sam_masks_numpy = sam_mask[0].detach().cpu().numpy()

      if len(sam_masks_numpy) == 0:
        print("No masks detected. Check model configuration or input image.")
        return None

      if show_masks:
        fig, axes = plt.subplots(1, 3, figsize=(20, 8)) 
        image_copy = image.copy()

        # Plot 4: SAM Masks
        axes[0].imshow(image)
        axes[0].set_title('Image')
        
        axes[1].imshow(image_copy)
        for i in range (len(boxes_numpy)):
          dataset_utils.show_box(boxes_numpy[i], axes[1])
        axes[2].imshow(image)
        dataset_utils.show_masks(sam_masks_numpy, axes[2])
        axes[2].set_title('Source Extractor with XAMI predicted masks')
        plt.axis('off')
        plt.grid(False)
        plt.tight_layout() 
        plt.savefig(f'./{image_path.split("/")[-1].replace(".png", "_predicted.png")}')
        plt.show()
        
      return sam_mask_pre, inference_time
    