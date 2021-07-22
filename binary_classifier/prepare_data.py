import os
import shutil
import numpy as np

HERE = os.path.abspath(__file__)
ROOT = os.path.dirname(os.path.dirname(HERE))
ORIG_DATA = os.path.join(ROOT, "gan-getting-started")
MONET_DIR = os.path.join(ORIG_DATA, "monet_jpg")
NON_MONET_DIR = os.path.join(ORIG_DATA, "photo_jpg")


def check_and_create(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def copy_files(files, dirname, stage_dir, file_type):

    for count, file in enumerate(files):
        source = os.path.join(dirname, file)
        target = os.path.join(stage_dir, f"{file_type}_{count}")
        shutil.copyfile(source, target)


def split_and_move(dirname):
    id = dirname.split("/")[-1].split("_")[0]

    files = os.listdir(dirname)

    num_files = len(files)

    np.random.seed(0)
    train, validate, test = np.split(files, [int(num_files * 0.7), int(num_files * 0.9)])

    data_dir = os.path.join(os.path.dirname(HERE), "data")
    check_and_create(data_dir)

    train_dir = os.path.join(data_dir, "train")
    validate_dir = os.path.join(data_dir, "validate")
    test_dir = os.path.join(data_dir, "test")

    stage_dirs = (train_dir, validate_dir, test_dir)
    stage_files = (train, validate, test)

    for d, f in zip(stage_dirs, stage_files):
        check_and_create(d)
        copy_files(f, dirname, d, id)


if __name__ == "__main__":
    split_and_move(MONET_DIR)
    split_and_move(NON_MONET_DIR)
