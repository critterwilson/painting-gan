import os
import shutil
import numpy as np
from typing import List

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
ORIG_DATA = os.path.join(ROOT, "gan-getting-started")
MONET_DIR = os.path.join(ORIG_DATA, "monet_jpg")
NON_MONET_DIR = os.path.join(ORIG_DATA, "photo_jpg")


def check_and_create(dirname: str) -> None:

    if not os.path.exists(dirname):
        os.mkdir(dirname)


def copy_files(files: List[str], dirname: str, stage_dir: str, file_type: str) -> None:

    for count, file in enumerate(files):
        source = os.path.join(dirname, file)
        target = os.path.join(stage_dir, f"{file_type}_{count}")
        shutil.copyfile(source, target)


def split_and_move(dirname: str) -> None:

    id = dirname.split("/")[-1].split("_")[0] # splitting gan-getting-started/monet_jpg -> monet

    files = os.listdir(dirname) 

    num_files = len(files)

    np.random.seed(0) # creates a rand numpy array?
    train, validate, test = np.split(files, [int(num_files * 0.7), int(num_files * 0.9)]) # splits the data into train/test/validate sets
                                                                                          # what are the parameters?
    data_dir = os.path.join(os.path.dirname(HERE), "data") # creates a new directory to store our data
    check_and_create(data_dir)

    train_dir = os.path.join(data_dir, "train") # creates sub directory for train
    validate_dir = os.path.join(data_dir, "validate") # " " for val
    test_dir = os.path.join(data_dir, "test") # " " for test

    stage_dirs = (train_dir, validate_dir, test_dir) # tuple of our directory names
    stage_files = (train, validate, test) # tuple of our filenames

    for d, f in zip(stage_dirs, stage_files): # copy the file over
        check_and_create(d)
        copy_files(f, dirname, d, id)


if __name__ == "__main__":

    split_and_move(MONET_DIR)
    split_and_move(NON_MONET_DIR)
