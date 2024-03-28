import json
import os
import shutil

class COCOToYOLOConverter:
    def __init__(self, input_path, output_path, annotations_file):
        self.input_path = input_path
        self.output_path = output_path
        self.train_data = None
        self.annotations_file = annotations_file
        self.file_names = []
        
        # Ensure output directories exist
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.output_path+'images/'):
            os.mkdir(self.output_path+'images/')
        if not os.path.exists(self.output_path+'labels/'):
            os.mkdir(self.output_path+'labels/')
    
    def load_images_from_folder(self, folder):
        count = 0
        filenames_from_json = list(set([file_['file_name'] for file_ in self.train_data['images']]))
        
        for filename in filenames_from_json:
            if filename.split('.')[-1] in ['jpg', 'jpeg', 'png']: 
                source = os.path.join(folder, filename)
                destination = f"{self.output_path}images/{filename}"
        
                try:
                    shutil.copy(source, destination)
                    # print(f"File {source} copied successfully.")
                except shutil.SameFileError:
                    print("Source and destination represent the same file.")
        
                self.file_names.append(filename)
                count += 1
        return count

    def get_img_ann(self, image_id):
        img_ann = []
        isFound = False
        for ann in self.train_data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        return img_ann if isFound else None

    def get_img(self, filename):
        for img in self.train_data['images']:
            if img['file_name'] == filename:
                return img

    def coco_to_yolo(self):
        count = 0
        for filename in self.file_names:
            img = self.get_img(filename)
            img_id = img['id']
            img_w = img['width']
            img_h = img['height']
        
            img_ann = self.get_img_ann(img_id)
            if img_ann is None:  # usually because the image doesn't have annotations
                continue

            with open(f"{self.output_path}labels/{'.'.join(filename.split('.')[:-1])}.txt", "a") as file_object:
                for ann in img_ann:
                    current_category = ann['category_id'] - 1  # Adjust if needed
                    current_bbox = ann['bbox']
                    x, y, w, h = current_bbox
                    
                    # Calculate and normalize midpoints
                    x_centre = (x + w / 2) / img_w
                    y_centre = (y + h / 2) / img_h
                    w = w / img_w
                    h = h / img_h
                    
                    # Write to file
                    file_object.write(f"{current_category} {x_centre:.6f} {y_centre:.6f} {w:.6f} {h:.6f}\n")
                
            count += 1

    def convert(self):
        input_json_train = os.path.join(self.input_path, self.annotations_file)
        with open(input_json_train) as f:
            self.train_data = json.load(f)

        self.load_images_from_folder(os.path.join(self.input_path, 'train/'))
        self.coco_to_yolo()
        print(f"Processed {len(self.file_names)} files.")