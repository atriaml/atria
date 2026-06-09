import numpy as np
from atria_logger import get_logger
from matplotlib.colors import LinearSegmentedColormap

logger = get_logger(__name__)

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30.0 / 255, 136.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((255.0 / 255, 13.0 / 255, 87.0 / 255, j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

colors = []
for j in np.linspace(1, 0, 100):
    colors.append((136.0 / 255, 30.0 / 255, 229.0 / 255, j))
for j in np.linspace(0, 1, 100):
    colors.append((13.0 / 255, 255.0 / 255, 87.0 / 255, j))
green_transparent_purple = LinearSegmentedColormap.from_list(
    "green_transparent_purple", colors
)


def score_to_color_map(explanation_score: float, color_map="red_transparent_blue"):
    if color_map == "red_transparent_blue":
        rgba = red_transparent_blue(explanation_score)
    elif color_map == "green_transparent_purple":
        rgba = green_transparent_purple(explanation_score)
    return rgba
