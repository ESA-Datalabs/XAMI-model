import cv2
from torch.utils.data import Dataset
from . import predictor_utils

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

        return predictor_utils.transform_image(self.model, self.transform, image, img_id, self.device)