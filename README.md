# PerspectiveNet-KITTI

## This code runs converts KITTI dataset into a format that is ready for processing by [detectron2](https://github.com/facebookresearch/detectron2)
The pipeline converts the dataset format from KITTI -> VOC -> COCO

### The instructions are as follows:

1. Create a kitti folder, coco folder and voc folder inside datasets
2. Create a folder called 'training' inside kitti
3. Place image_2 and label_2 folders inside of 'training'
4. Run the script as python3 utils.py --r t
