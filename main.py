import argparse
import json
import os
import tempfile
from typing import (Optional, cast, )

import cv2  # type: ignore
import pyheif  # type: ignore
import numpy as np
from scipy.stats import vonmises  # type: ignore
from PIL import Image  # type: ignore
from numpy.typing import (NDArray, )
from matplotlib import pyplot as plt  # type: ignore

SATURATION_WHITE: np.uint8 = np.uint8(4)
SATURATION_BRIGHT_WHITE: np.uint8 = np.uint8(0)

VALUE_BACKGROUND: np.uint8 = np.uint8(16)
VALUE_BLACK: np.uint8 = np.uint8(8)
VALUE_WHITE: np.uint8 = 256 - np.uint8(32)
VALUE_BRIGHT_BLACK: np.uint8 = np.uint8(96)
VALUE_BRIGHT_WHITE: np.uint8 = np.uint8(255)

MINIMUM_VALUE_COLOR: np.uint8 = np.uint8(min(VALUE_BACKGROUND * 3 * 3.5, 255))
BRIGHT_RATIO: float = 1.1

Radian = np.float64


def get_average(
    numbers: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
) -> float:
    if weights is None:
        weights = np.full_like(numbers, 1)
    sum_weights = np.sum(weights)
    return np.sum(numbers.astype(np.float64) * weights) / sum_weights


def get_circular_average(
    radians: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None
) -> np.float64:

    if weights is None:
        weights = np.full_like(radians, 1)

    sum_sin = np.sum(np.sin(radians) * weights)
    sum_cos = np.sum(np.cos(radians) * weights)

    circular_average = np.arctan2(sum_sin, sum_cos)
    return circular_average


def get_circular_variance(
    radians: NDArray[np.float64],
    weights: Optional[NDArray[np.float64]] = None,
    average: Optional[np.float64] = None,
) -> np.float64:

    if weights is None:
        weights = np.full_like(radians, 1)

    if average is None:
        average = get_circular_average(radians, weights)

    weighted_average = np.average(np.cos(radians - average), weights=weights)
    circular_variance = 1 - weighted_average
    return circular_variance


def get_hue_weight(
    image_hue: NDArray[Radian],
    target_hue: Radian,
    *,
    factor: float = 1.0
) -> NDArray[np.float64]:
    stddev = np.pi / 6 * factor
    kappa = 1 / np.power(stddev, 2)
    return np.array([vonmises.pdf(e, kappa, target_hue) for e in image_hue])


def get_palette_hue(
    image_hue: NDArray[Radian],
    target_hue: Radian,
    non_hue_weight: NDArray[np.float64],
    *,
    alpha: float
) -> Radian:
    hue_weight = get_hue_weight(image_hue, target_hue, factor=alpha)
    weight_for_palette_hue = hue_weight * non_hue_weight
    palette_hue = get_circular_average(image_hue, weight_for_palette_hue)
    hue_difference = palette_hue - target_hue
    trim_boundary = np.pi / 12
    if trim_boundary < hue_difference and hue_difference <= np.pi / 2:
        palette_hue = (target_hue + trim_boundary) % np.pi
    elif np.pi / 2 < hue_difference and hue_difference < np.pi - trim_boundary:
        palette_hue = (target_hue + np.pi - trim_boundary) % np.pi
    return palette_hue


def get_palette_color(
    image: NDArray[np.uint8],
    palette_hue: Radian,
    non_hue_weight: NDArray[np.float64],
    *,
    beta: float
) -> NDArray[np.uint8]:
    _hue: np.uint8
    saturation: np.uint8
    value: np.uint8
    _hue, saturation, value = cv2.split(image)
    hue = np.radians(_hue, dtype=Radian)

    palette_hue_weights = get_hue_weight(hue, palette_hue, factor=beta)
    weights = palette_hue_weights * non_hue_weight

    average_saturation = np.average(saturation, weights=weights)
    average_value = np.average(value, weights=weights)

    _hue = np.uint8(np.degrees(palette_hue))
    color = np.array([_hue, average_saturation, average_value], dtype=np.uint8)
    return color


def hsv_to_hex(color: NDArray[np.uint8]) -> str:
    color_rgb = cv2.cvtColor(
        color.reshape((1, 1, 3)), cv2.COLOR_HSV2RGB)
    r, g, b = tuple(color_rgb[0][0])
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color_hex


def normalize_to_target_value(
    color: NDArray[np.uint8],
    target_value: np.uint8
) -> NDArray[np.uint8]:
    hue, saturation, value = color[0], color[1], color[2]
    normalized_saturation = cast(np.float64, saturation) / 255
    normalized_value = cast(np.float64, value) / 255
    factor = int(normalized_saturation * normalized_value * 255)

    target_saturation = np.uint8(min(factor/int(target_value), 1) * 255)
    normalized_color = np.array(
        [hue, target_saturation, target_value], dtype=np.uint8)
    return normalized_color


def get_grey_colors(
    image_hue: NDArray[np.float64],
    primary_hue: np.float64,
    non_hue_weight: NDArray[np.float64],
    values: list[np.uint8],
    *,
    gamma: float
) -> list[NDArray[np.uint8]]:

    hue_weight = get_hue_weight(image_hue, primary_hue)
    factor = int(np.average(non_hue_weight, weights=hue_weight) * gamma * 255)

    color = np.array([np.degrees(primary_hue), 255, factor], np.uint8)
    return [normalize_to_target_value(color, value) for value in values]


def read_heic(filename: str) -> cv2.typing.MatLike:
    _, tempname = tempfile.mkstemp(suffix='.jpg')
    heif_file = pyheif.read(filename)
    pil_image = Image.frombytes(
        heif_file.mode, heif_file.size, heif_file.data,
        "raw", heif_file.mode, heif_file.stride
    )
    pil_image.save(tempname)
    image = cv2.imread(tempname)
    os.remove(tempname)
    return image


def main(args: argparse.Namespace):
    debug: bool = args.debug

    filename = os.path.expanduser(args.filename)
    output = os.path.expanduser(args.output)
    if os.path.splitext(filename)[1] == '.heic':
        image = read_heic(filename)
    else:
        image = cv2.imread(filename)
    target_size = 1280 * 1024
    factor = target_size / image.size
    if factor < 1:
        image = cv2.resize(image, dsize=(0, 0), fx=factor, fy=factor)
    assert image is not None

    alpha: float = args.alpha
    beta: float = args.beta
    gamma: float = args.gamma

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _hue: NDArray[np.uint8]
    saturation: NDArray[np.uint8]
    value: NDArray[np.uint8]
    _hue, saturation, value = cv2.split(image_hsv)
    hue: NDArray[np.float64] = np.radians(_hue, dtype=np.float64)

    palette: NDArray[np.uint8] = np.zeros((2, 8, 3), dtype=np.uint8)
    theme: dict[str, dict[str, str]] = {"special": {}, "colors": {}}

    saturation_weight = saturation.astype(np.float64) / 255
    value_weight = value.astype(np.float64) / 255
    non_hue_weight = saturation_weight * value_weight

    primary_hue = get_circular_average(hue, non_hue_weight)
    values = [
        VALUE_BACKGROUND,
        VALUE_BLACK,
        VALUE_BRIGHT_BLACK,
    ]
    grey_colors = get_grey_colors(
        hue, primary_hue, non_hue_weight, values, gamma=gamma)
    white = np.array([primary_hue, SATURATION_WHITE, VALUE_WHITE], np.uint8)
    bright_white = np.array([0, 0, VALUE_BRIGHT_WHITE], np.uint8)

    theme['special']['background'] = hsv_to_hex(grey_colors[0])
    theme['special']['foreground'] = hsv_to_hex(white)
    theme['special']['cursor'] = hsv_to_hex(white)
    palette[0][0] = grey_colors[1]
    palette[0][7] = white
    palette[1][0] = grey_colors[2]
    palette[1][7] = bright_white
    theme['colors']['color0'] = hsv_to_hex(grey_colors[1])
    theme['colors']['color7'] = hsv_to_hex(white)
    theme['colors']['color8'] = hsv_to_hex(grey_colors[2])
    theme['colors']['color15'] = hsv_to_hex(bright_white)

    target_hues = np.radians([0, 60, 30, 120, 150, 90], dtype=Radian)
    for i, target_hue in enumerate(target_hues, 1):
        palette_hue = get_palette_hue(
            hue, target_hue, non_hue_weight, alpha=alpha)
        color = get_palette_color(
            image_hsv, palette_hue, non_hue_weight, beta=beta)
        _value: np.uint8 = color[2]
        if _value < MINIMUM_VALUE_COLOR:
            color = normalize_to_target_value(color, MINIMUM_VALUE_COLOR)
            _value = MINIMUM_VALUE_COLOR
        bright_color = normalize_to_target_value(
            color, np.uint8(min(255, float(_value) * BRIGHT_RATIO)))

        theme['colors'][f'color{i}'] = hsv_to_hex(color)
        theme['colors'][f'color{i+8}'] = hsv_to_hex(bright_color)
        palette[0][i] = color
        palette[1][i] = bright_color

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(os.path.expanduser(output), 'w') as f:
        json.dump(theme, f)

    if debug:
        plt.subplot(2, 2, 1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        plt.subplot(2, 2, 2)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(palette, cv2.COLOR_HSV2RGB))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename',
        default='./resources/tulips.png',
        help=("(default: ./resources/tuplips.png")
    )
    parser.add_argument(
        '--output',
        default="~/.config/wal/colorschemes/dark/custom.json",
        help=("(default: ~/.config/wal/colorschemes/dark/custom.json)")
    )
    parser.add_argument(
        '--alpha',
        default=1.0,
        help=(
            "A stddev factor for the hue of palette color, "
            "which is of the weight of colors around the primary hue"
        )
    )
    parser.add_argument(
        '--beta',
        default=1.0,
        help=(
            "A stddev factor for the value and saturation of palette color, "
            "which is of the weight of colors around the palette hue"
        )
    )
    parser.add_argument(
        '--gamma',
        default=0.5,
        help=(
            "A weight for the saturations "
            "of background, black and bright black colors"
        )
    )
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)
