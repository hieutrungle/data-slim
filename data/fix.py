from PIL import Image, ImageChops
import os
import glob
import shutil
import pathlib
import sys


def is_grayscale(image):
    """Check if image is monochrome (1 channel or 3 identical channels)"""
    if image.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if image.mode == "RGB":
        rgb = image.split()
        if ImageChops.difference(rgb[0], rgb[1]).getextrema()[1] != 0:
            return False
        if ImageChops.difference(rgb[0], rgb[2]).getextrema()[1] != 0:
            return False
    return True


def remove_grayscale(files):
    for file in files:
        try:
            img = Image.open(file)
            if is_grayscale(img):
                print(file)
                os.remove(file)
        except KeyboardInterrupt:
            print("Keyboard interrupt, stopping")
            sys.exit(0)
        except:
            os.remove(file)


def move_grayscale(files):
    for file in files:
        try:
            img = Image.open(file)
            if is_grayscale(img):
                print(file)
                name = file.partition("train/")[-1]
                gray_folder = "gray"
                new_dest = os.path.join(pathlib.Path.cwd(), gray_folder, name)
                shutil.move(file, new_dest)
        except KeyboardInterrupt:
            print("Keyboard interrupt, stopping")
            sys.exit(0)
        except ValueError:
            print(f"ValueError for {file}")
            name = file.partition("train/")[-1]
            gray_folder = "gray"
            new_dest = os.path.join(pathlib.Path.cwd(), gray_folder, name)
            shutil.move(file, new_dest)
        except:
            print(f"Unexpected exception while moving image: {file}")
            name = file.partition("train/")[-1]
            gray_folder = "gray"
            new_dest = os.path.join(pathlib.Path.cwd(), gray_folder, name)
            shutil.move(file, new_dest)


if __name__ == "__main__":
    files = glob.glob(os.path.join("./train/", '*.png'))
    move_grayscale(files)
    files = glob.glob(os.path.join("./train/", '*.png'))
    remove_grayscale(files)
