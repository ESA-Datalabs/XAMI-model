# XAMI model training


### Train YOLOv8 layers

Check [run_yolo_train_splits.sh](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/run_yolo_train_splits.sh) and [train_detector.py](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/train_detector.py) on how to train the YOLO model. 

```bash
./run_yolo_train_splits.sh
```

### Train SAM layers

Check [run_sam_train_splits.sh](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/run_sam_train_splits.sh) and [train_sam.py](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/train_sam.py)

```bash
./run_sam_train_splits.sh
```

### Train YOLOv8 and SAM together

See [train_yolo_sam.ipynb](https://github.com/ESA-Datalabs/XAMI-model/blob/main/train/train_yolo_sam.ipynb) for how to couple the models to train SAM layers with predicted YOLO bounding boxes and compute metrics. 