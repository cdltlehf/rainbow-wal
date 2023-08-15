import argparse
import json
import os
from typing import (Optional, Union, cast, )

import cv2
import numpy as np
from numpy.typing import (NDArray, )


def get_average(
    numbers: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
) -> float:
    if weights is None:
        weights = np.full_like(numbers, 1)
    sum_weights = np.sum(weights)
    return np.sum(numbers.astype(np.float64) * weights) / sum_weights


def get_circular_average(
    numbers: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None
) -> np.ndarray:

    if weights is None:
        weights = np.full_like(numbers, 1)

    radians = np.array(numbers) * np.pi * 2

    sum_sin = np.sum(np.sin(radians) * weights)
    sum_cos = np.sum(np.cos(radians) * weights)

    average_radian = np.arctan2(sum_sin, sum_cos)
    circular_average = average_radian / (2 * np.pi)
    if circular_average < 0:
        circular_average += 1
    if circular_average >= 1:
        circular_average -= 1
    return circular_average


def get_average_hue(
    hues: NDArray[np.uint8],
    weights: Optional[NDArray[np.float64]] = None,
) -> np.uint8:
    normalized_hues = hues.flatten().astype(np.float64) / 180
    if weights is not None:
        weights = weights.flatten()
    return np.uint8(get_circular_average(normalized_hues, weights) * 180)


def get_hue_similarity(
    hue1: NDArray[np.uint8],
    hue2: Union[NDArray[np.uint8], np.uint8],
) -> NDArray[np.float64]:
    hue1_int32 = hue1.astype(np.int32)
    hue2_int32: NDArray[np.int32]
    if np.isscalar(hue2):
        hue2_int32 = cast(NDArray[np.int32], hue2)
    else:
        hue2_int32 = cast(NDArray[np.int32], hue2).astype(np.int32)

    distances = (hue1_int32 - hue2_int32 + 180) % 180
    distances[distances > 90] = 180 - distances[distances > 90]
    return (1 - distances.astype(np.float64) / 90) * 0.99


def get_primary_hue(
    image_hue: NDArray[np.uint8],
    target_hue: np.uint8,
    non_hue_weight: NDArray[np.float64],
    *,
    alpha: float = 10
) -> np.uint8:

    hue_weight = get_hue_similarity(image_hue, target_hue)
    hue_weight = np.power(hue_weight, alpha)
    weight_for_primary_hue = hue_weight * non_hue_weight
    primary_hue = get_average_hue(image_hue, weight_for_primary_hue)

    return primary_hue


def get_primary_color(
    image: NDArray[np.uint8],
    primary_hue: np.uint8,
    non_hue_weight: NDArray[np.float64],
    *,
    beta: float = 10,
) -> NDArray[np.uint8]:
    hue, saturation, value = cv2.split(image)

    primary_hue_weight = get_hue_similarity(hue, primary_hue)

    weight = primary_hue_weight * np.power(non_hue_weight, beta)

    average_saturation = get_average(saturation.flatten(), weight.flatten())
    average_value = get_average(value.flatten(), weight.flatten())

    color = np.array(
        [primary_hue, average_saturation, average_value], np.uint8)
    return color


def hsv_to_hex(color: NDArray[np.uint8]) -> str:
    color_rgb = cv2.cvtColor(
        color.reshape((1, 1, 3)), cv2.COLOR_HSV2RGB)
    r, g, b = tuple(color_rgb[0][0])
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color_hex


def get_grey_colors(
    image_hue: NDArray[np.uint8],
    primary_hue: np.uint8,
    non_hue_weight: NDArray[np.float64],
    values: list[np.uint8],
    *,
    gamma: float = 4,
) -> list[NDArray[np.uint8]]:

    normalized_values = [cast(np.float64, value) / 255 for value in values]
    hue_weight = get_hue_similarity(image_hue, primary_hue)
    factor = get_average(non_hue_weight, hue_weight) / gamma

    saturations = [
        min(float(factor/value), 1) * 255
        for value in normalized_values
    ]
    return [
        np.array([primary_hue, saturation, value], np.uint8)
        for saturation, value in zip(saturations, values)
    ]


def main(args: argparse.Namespace):

    theme = {
        "special": {
            "background": "#000000",
            "foreground": "#f5f5f7",
            "cursor": "#f5f5f7"
        },
        "colors": {f"color{i}": "#ffffff" for i in range(16)}
    }
    theme["colors"]["color0"] = "#000000"
    theme["colors"]["color7"] = "#ffffff"
    theme["colors"]["color8"] = "#555555"
    theme["colors"]["color15"] = "#ffffff"

    filename = os.path.expanduser(args.filename)
    image = cv2.imread(filename)
    # img = cv2.resize(img, dsize=(0, 0), fx=0.1, fy=0.1)
    cv2.imshow('original', image)
    assert image is not None

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(image_hsv)

    saturation_weight = saturation.astype(np.float64) / 255
    value_weight = value.astype(np.float64) / 255
    non_hue_weight = saturation_weight * value_weight

    primary_hue = get_average_hue(hue, non_hue_weight)
    values = cast(list[np.uint8], [24, 8, 128])
    grey_colors = get_grey_colors(hue, primary_hue, non_hue_weight, values)

    theme['special']['background'] = hsv_to_hex(grey_colors[0])
    theme['colors']['color0'] = hsv_to_hex(grey_colors[1])
    theme['colors']['color8'] = hsv_to_hex(grey_colors[2])

    target_hues = cast(list[np.uint8], [0, 60, 30, 120, 150, 90])
    for i, target_hue in enumerate(target_hues, 1):
        primary_hue = get_primary_hue(hue, target_hue, non_hue_weight)
        color = get_primary_color(image_hsv, primary_hue, non_hue_weight)
        color_bright = get_primary_color(
            image_hsv, primary_hue, non_hue_weight, beta=100
        )
        theme['colors'][f'color{i}'] = hsv_to_hex(color)
        theme['colors'][f'color{i+8}'] = hsv_to_hex(color_bright)

    output_path = '~/.config/wal/colorschemes/dark/custom.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(os.path.expanduser(output_path), 'w') as f:
        json.dump(theme, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        default="~/dotfiles/wallpapers/Wallpaper-Orsay.default.jpg"
    )

    args = parser.parse_args()
    main(args)
