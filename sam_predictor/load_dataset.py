import cv2
from torch.utils.data import Dataset
import torch
from torchvision.transforms.functional import resize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2

class ImageDataset(Dataset):
    def __init__(self, image_paths, model, transform, device):
        self.image_paths = image_paths
        self.model = model
        self.transform = transform
        self.device = device
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_id = self.image_paths[idx].split("/")[-1]
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        
        gray_image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Apply a colormap
        # rgb_image = self.apply_colormap(gray_image, colormap_name='turbo')
        rgb_image = plt.cm.turbo(gray_image / np.max(gray_image))
        rgb_image = (rgb_image * 255).astype(np.uint8)[:, :, :3]

        return self.transform_image(self.model, self.transform, image, img_id, self.device)
        
    def transform_image(self, model, transform, image, k, device):
        
            # sets a specific mean for each image
            image_T = np.transpose(image, (2, 1, 0))
            mean_ = np.mean(image_T[image_T>0])
            std_ = np.std(image_T[image_T>0]) 
            pixel_mean = torch.as_tensor([mean_, mean_, mean_], dtype=torch.float, device=device)
            pixel_std = torch.as_tensor([std_, std_, std_], dtype=torch.float, device=device)
            # print(pixel_mean, pixel_std)
            # plt.imshow(image)
            # plt.show()
            # plt.close()
            model.register_buffer("pixel_mean", torch.Tensor(pixel_mean).unsqueeze(-1).unsqueeze(-1), False) # not in SAM
            model.register_buffer("pixel_std", torch.Tensor(pixel_std).unsqueeze(-1).unsqueeze(-1), False) # not in SAM

            transformed_data = {}
            negative_mask = np.where(image > 0, True, False)
            negative_mask = torch.from_numpy(negative_mask)  
            negative_mask = negative_mask.permute(2, 0, 1)
            negative_mask = resize(negative_mask, [1024, 1024], antialias=True) 
            negative_mask = negative_mask.unsqueeze(0)
            # scales the image to 1024x1024 by longest side 
            input_image = transform.apply_image(image)
            input_image_torch = torch.as_tensor(input_image, device=device)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            
            # normalization and padding
            input_image = model.preprocess(transformed_image)
            original_image_size = image.shape[:2]
            input_size = tuple(transformed_image.shape[-2:])
            input_image[~negative_mask] = 0
            transformed_data['image'] = input_image.clone() 
            transformed_data['input_size'] = input_size 
            transformed_data['image_id'] = k
            transformed_data['original_image_size'] = original_image_size
        
            return transformed_data
        
    def apply_colormap(self, gray_image, colormap_name='inferno'):
        # Normalize the image to range between 0 and 1
        normalized_image = mcolors.Normalize(vmin=gray_image.min(), vmax=gray_image.max())(gray_image)
        
        mapped_color_image = plt.cm.get_cmap(colormap_name)(normalized_image)
        
        # Convert to RGB format by dropping the alpha channel and multiplying by 255
        color_image = (mapped_color_image[:, :, :3] * 255).astype('uint8')
        
        return color_image
            
                