# %%
import numpy as np
from src.read_dir_images import ImgsInDirAsBool, ImgsInDirAsGray
import cv2
from IPython.display import display, Image

# %%


def display_cv(image, format='.bmp'):
    decoded_bytes = cv2.imencode(format, image)[1].tobytes()
    display(Image(data=decoded_bytes))


# %%
fish_files = ImgsInDirAsGray('./images/pin/fish')
ref_files = ImgsInDirAsBool('./images/pin/ref',  bool_switch=True)
true_files = ImgsInDirAsBool('./images/pin/true', bool_switch=True)
false_files = ImgsInDirAsBool('./images/pin/false', bool_switch=True)

# %%
for num, (fish, ref, true, false) in enumerate(
        zip(fish_files.read_files(), ref_files.read_files(),
            true_files.read_files(), false_files.read_files())):
    pass


# %%


def measure_image_area(true: np.ndarray, false: np.ndarray):
    pass

# %%


def label_connected_area(img: np.ndarray):
    label = np.zeros_like(img)
