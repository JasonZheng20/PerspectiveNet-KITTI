# ============================================================================ #
# Utils.py
#
# Provides utilities and data preprocessing on the KITTI dataset for 231N Final
# Project. Transforms data from KITTI CSV labels and monocular images to VOC
# format, and from VOC to COCO.
# Jason Zheng
# 5/15/20
# ============================================================================ #

import argparse
import time
import torch

from math import ceil, floor
from numpy.random import choice
from os import listdir
from os.path import isfile, join
from umautobots_converter.main import main as convert

# ============================================================================ #
def partition_kitti_datasets():
    """
    @params: []
    @returns: []
    Partitions the dataset and labels without assuming a complete, continuous
    dataset by title number. Saves these partitionings in text files at path
    './datasets/'

    Note: We have our own partitioning function because we are missing image
    indices, and thus cannot use train_mapping/train_rand in devkit_object.
    Furthermore, it seems according to the KITTI website that student projects
    cannot access the test split of the dataset and must instead create their
    own training/val/test split from the training data.
    """
    print('Partitioning KITTI Dataset ...')
    initialTime = time.time()
    dataset_path = join(args.kitti_dataset_path, "training/image_2")  # Get the fragmented image folder
    sample_list = []
    num_samples = 0
    for file in listdir(dataset_path):
        if not isfile(join(dataset_path, file)):
            continue
        sample_index = file[:-1*len(args.photo_format)-1]
        sample_list.append(sample_index)
        num_samples += 1
    num_train_samples = ceil(num_samples*args.dataset_split_ratio[0])
    num_test_samples = ceil(num_samples*args.dataset_split_ratio[2])
    num_val_samples = num_samples - num_train_samples - num_test_samples
    train = choice(sample_list, num_train_samples, replace=False)
    test = choice(sample_list, num_test_samples, replace=False)
    val = choice(sample_list, num_val_samples, replace=False)
    with open(join(args.kitti_dataset_path,"train.txt"), "w") as train_file:
        for train_sample in train:
            train_file.write("%s\n"%train_sample)
    with open(join(args.kitti_dataset_path,"test.txt"), "w") as test_file:
        for test_sample in test:
            test_file.write("%s\n"%test_sample)
    with open(join(args.kitti_dataset_path,"val.txt"), "w") as val_file:
        for val_sample in val:
            val_file.write("%s\n"%val_sample)
    print('Found %d samples. Partitioned into %d train, %d val and %d test in %2f s' % (
        num_samples, num_train_samples, num_val_samples, num_test_samples, time.time() - initialTime
        )
    )
    return train, val, test

def to_voc():
    """
    @params: []
    @returns: []
    Converts the dataset from KITTI format to VOC Format
    """
    initialTime = time.time()

    print('Validating splits ...')
    if not validate_splits():
        # Get the splits if they do not exist. We must redo the entire partition
        # in order to avoid repeated indices across splits
        print('Re-partitioning to obtain valid splits ...')
        partition_kitti_datasets()

    # Run the conversion script from umautobots KITTI to VOC converter
    # https://github.com/umautobots/vod-converter
    for split in ['train', 'val', 'test']:
        print('Converting %s split from kitti to voc ...' % split)
        convertInitialTime = time.time()
        convert(
            from_path=args.kitti_dataset_path,
            from_key='kitti',
            to_path=args.voc_dataset_path,
            to_key='voc',
            split=split
        )
        print('Converted %s split in %2f s' % (split, time.time() - convertInitialTime))
    print('Completed kitti to voc conversion in %2f s' % (time.time() - initialTime))

def to_coco():
    """
    @params: []
    @returns: []
    Converts the dataset from VOC Format to COCO Format
    """
    pass

def validate_splits():
    """
    @params: []
    @returns: (bool) True if valid split exists, False otherwise
    Checks that valid splits exist at the kitti dataset path
    """
    files = [file for file in listdir(args.kitti_dataset_path) if isfile(join(args.kitti_dataset_path, file))]
    if 'train.txt' not in files or 'test.txt' not in files or 'val.txt' not in files:
        print('Could not find a valid split in the provided KITTI dataset directory')
        return False
    return True


# ============================================================================ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set flag to 'v' for CSV->VOC, 'c' for VOC-> COCO, 't' for CSV->VOC->COCO, 'p' for partition
    parser.add_argument('--r', type=str, help='Dataset transform to run')

    # Partition arguments
    parser.add_argument('--photo_format', type=str, default='png', help='Format of image files')
    parser.add_argument('--dataset_split_ratio', nargs=3, type=float, default=[0.85, 0.05, 0.1], help='Train/Val/Test split')

    # Conversion arguments
    parser.add_argument('--kitti_dataset_path', type=str, default='./datasets/kitti/', help='Base KITTI Dataset path')
    parser.add_argument('--voc_dataset_path', type=str, default='./datasets/voc/', help='Base VOC Dataset path')
    parser.add_argument('--coco_dataset_path', type=str, default='./datasets/coco', help='Base COCO Dataset path')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.r == 'p' or args.r == 't':
        train, val, test = partition_kitti_datasets()
    if args.r == 'v' or args.r == 't':
        to_voc()
    if args.r == 'c' or args.r == 't':
        to_coco()
