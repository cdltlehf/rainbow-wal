import argparse
import json
import os
import tempfile
from typing import (Optional, )

import cv2  # type: ignore
import pyheif  # type: ignore
import numpy as np
from scipy.stats import vonmises  # type: ignore
from PIL import Image  # type: ignore
from numpy.typing import (NDArray, )
from matplotlib import pyplot as plt  # type: ignore

VALUE_BACKGROUND: np.uint8 = np.uint8(16)
VALUE_BLACK: np.uint8 = np.uint8(8)
VALUE_BRIGHT_BLACK: np.uint8 = np.uint8(96)

MINIMUM_VALUE_CHROMATIC_COLOR: np.uint8 = (
    np.uint8(min((VALUE_BACKGROUND + 8) * 7 - 8, 255)))
BRIGHT_RATIO: float = 1.1

MAXIMUM_IMAGE_SIZE: int = 1280 * 1024
CHROMATIC_COLOR_WEIGHT_THRESHOLD: float = 1 / 128

Radian = np.float64


def get_default_palette() -> NDArray[np.uint8]:
    # https://developer.apple.com/design/human-interface-guidelines/color#macOS-system-colors
    palette = np.zeros((2, 8, 3), dtype=np.uint8)

    palette[0][1] = np.array([255, 69, 58])
    palette[0][2] = np.array([50, 215, 75])
    palette[0][3] = np.array([255, 214, 10])
    palette[0][4] = np.array([10, 132, 255])
    palette[0][5] = np.array([255, 55, 95])
    palette[0][6] = np.array([90, 200, 245])
    palette[0][7] = np.array([240, 240, 240])

    palette[1][1] = np.array([255, 105, 97])
    palette[1][2] = np.array([49, 222, 75])
    palette[1][3] = np.array([255, 212, 38])
    palette[1][4] = np.array([64, 156, 255])
    palette[1][5] = np.array([255, 100, 130])
    palette[1][6] = np.array([112, 215, 255])
    palette[1][7] = np.array([255, 255, 255])

    return palette


def hsv_to_rgb(color: NDArray[np.uint8]) -> NDArray[np.uint8]:
    color_rgb = cv2.cvtColor(color.reshape((1, 1, 3)), cv2.COLOR_HSV2RGB)
    return color_rgb[0][0]


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


def get_hue_weights(
    image_hues: NDArray[Radian],
    target_hue: Radian,
    *,
    factor: float = 1.0
) -> NDArray[np.float64]:
    stddev = np.pi / 6 / factor
    kappa = 1 / np.power(stddev, 2)
    return np.array([vonmises.pdf(e, kappa, target_hue) for e in image_hues])


def get_hue_chromatic_color(
    image_hues: NDArray[Radian],
    target_hue: Radian,
    chromas: NDArray[np.float64],
    *,
    alpha: float
) -> Radian:
    hue_weights = get_hue_weights(image_hues, target_hue, factor=alpha)
    weights = hue_weights * chromas
    if np.average(weights) < CHROMATIC_COLOR_WEIGHT_THRESHOLD:
        return target_hue
    hue_chromatic_color = get_circular_average(image_hues, weights)

    # -np.pi < hue_difference < np.pi
    hue_difference = hue_chromatic_color - target_hue
    if hue_difference > np.pi:
        hue_difference -= np.pi
    elif hue_difference < -np.pi:
        hue_difference += np.pi

    trim_boundary = np.pi / 12
    if hue_difference > trim_boundary:
        hue_chromatic_color = target_hue + trim_boundary
    elif hue_difference < -trim_boundary:
        hue_chromatic_color = target_hue - trim_boundary

    # 0 <= hue_chromatic_color < 2 * pi
    if hue_chromatic_color < 0:
        hue_chromatic_color += 2 * np.pi
    elif hue_chromatic_color > 2 * np.pi:
        hue_chromatic_color -= 2 * np.pi

    return hue_chromatic_color


def get_chromatic_color(
    image_hsv: tuple[
        NDArray[np.float64],
        NDArray[np.uint8],
        NDArray[np.uint8]
    ],
    hue_chromatic_color: Radian,
    chromas: NDArray[np.float64],
    primary_hue_weights: NDArray[np.float64],
    *,
    beta: float
) -> Optional[NDArray[np.uint8]]:

    image_hues, image_saturations, image_values = image_hsv
    _hue = np.uint8(np.degrees(hue_chromatic_color))

    hue_weights = get_hue_weights(image_hues, hue_chromatic_color, factor=beta)
    weights = hue_weights * chromas
    if np.average(weights) < CHROMATIC_COLOR_WEIGHT_THRESHOLD:
        weights = primary_hue_weights * chromas
    if np.average(weights) < CHROMATIC_COLOR_WEIGHT_THRESHOLD:
        return None
    _saturation = np.average(image_saturations, weights=weights)
    _value = np.average(image_values, weights=weights)
    _hue = np.uint8(np.degrees(hue_chromatic_color))
    color = np.array([_hue, _saturation, _value], dtype=np.uint8)
    return color


def rgb_to_hex(color: NDArray[np.uint8]) -> str:
    r, g, b = tuple(color.tolist())
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color_hex


def standardize_with_value(
    color: NDArray[np.uint8],
    target_value: np.uint8
) -> NDArray[np.uint8]:
    hue, saturation, value = color[0], color[1], color[2]

    normalized_saturation = saturation.astype(np.float64) / 255
    normalized_value = value.astype(np.float64) / 255
    chroma = normalized_saturation * normalized_value

    normalized_target_value = float(target_value) / 255
    normalized_target_saturation = min(chroma / normalized_target_value, 1)
    target_saturation = np.uint8(normalized_target_saturation * 255)

    standardized_color = np.array(
        [hue, target_saturation, target_value], dtype=np.uint8)
    return standardized_color


def get_achromatic_colors(
    primary_hue: np.float64,
    primary_hue_weights: NDArray[np.float64],
    chromas: NDArray[np.float64],
    values: list[np.uint8],
    *,
    gamma: float
) -> list[NDArray[np.uint8]]:

    average_chroma = np.average(chromas, weights=primary_hue_weights)
    _value = int(min(float(average_chroma) / gamma, 1) * 255)

    color = np.array([np.degrees(primary_hue), 255, _value], np.uint8)
    return [standardize_with_value(color, value) for value in values]


def read_heic(filename: str):
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


def get_chromas(
    saturation: NDArray[np.uint8],
    value: NDArray[np.uint8],
) -> NDArray[np.float64]:
    normalized_saturation = saturation.astype(np.float64) / 255
    normalized_value = value.astype(np.float64) / 255
    chroma = normalized_saturation * normalized_value
    return chroma


def main(args: argparse.Namespace) -> None:
    debug: bool = args.debug

    filename = os.path.expanduser(args.filename)
    output = os.path.expanduser(args.output)
    if os.path.splitext(filename)[1] == '.heic':
        image = read_heic(filename)
    else:
        image = cv2.imread(filename)
    factor = np.sqrt(MAXIMUM_IMAGE_SIZE / image.size)
    if factor < 1:
        resized_image = cv2.resize(image, dsize=(0, 0), fx=factor, fy=factor)
    else:
        resized_image = image
    assert resized_image is not None

    alpha: float = args.alpha
    beta: float = args.beta
    gamma: float = args.gamma

    _image_hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
    _image_hues, image_saturations, image_values = cv2.split(_image_hsv)
    image_hues: NDArray[np.float64] = np.radians(_image_hues, dtype=np.float64)
    image_hsv = (image_hues, image_saturations, image_values)

    background_color: NDArray[np.uint8]
    palette = get_default_palette()
    theme: dict[str, dict[str, str]] = {"special": {}, "colors": {}}

    chromas = get_chromas(image_saturations, image_values)
    primary_hue = get_circular_average(image_hues, chromas)
    values_achromatic_color = [
        VALUE_BACKGROUND, VALUE_BLACK, VALUE_BRIGHT_BLACK
    ]
    primary_hue_weights = get_hue_weights(image_hues, primary_hue)
    achromatic_colors = get_achromatic_colors(
        primary_hue, primary_hue_weights, chromas, values_achromatic_color,
        gamma=gamma
    )

    background_color = hsv_to_rgb(achromatic_colors[0])
    palette[0][0] = hsv_to_rgb(achromatic_colors[1])
    palette[1][0] = hsv_to_rgb(achromatic_colors[2])

    target_hues = np.radians([0, 60, 30, 120, 150, 90], dtype=Radian)
    for i, target_hue in enumerate(target_hues, 1):
        hue_chromatic_color = get_hue_chromatic_color(
            image_hues, target_hue, chromas, alpha=alpha)
        color = get_chromatic_color(
            image_hsv, hue_chromatic_color, chromas, primary_hue_weights,
            beta=beta
        )
        if color is None:
            continue

        value_chromatic_color = max(color[2], MINIMUM_VALUE_CHROMATIC_COLOR)
        value_bright_chromatic_color = min(
            value_chromatic_color * BRIGHT_RATIO, 255)

        color = standardize_with_value(color, value_chromatic_color)
        bright_color = standardize_with_value(
            color, value_bright_chromatic_color)

        palette[0][i] = hsv_to_rgb(color)
        palette[1][i] = hsv_to_rgb(bright_color)

    theme['special']['background'] = rgb_to_hex(background_color)
    theme['special']['foreground'] = rgb_to_hex(palette[0][7])
    theme['special']['cursor'] = rgb_to_hex(palette[0][7])
    for i in range(2):
        for j in range(8):
            theme['colors'][f'color{8 * i + j}'] = rgb_to_hex(palette[i][j])

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(os.path.expanduser(output), 'w') as f:
        json.dump(theme, f)

    if debug:
        plt.subplot(2, 1, 1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

        plt.subplot(2, 1, 2)
        plt.axis("off")
        plt.imshow(palette)
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
        type=float,
        default=2,
        help=(
            "A inverse-stddev factor for the hue of the palette, "
            "which is of the weight of colors around the primary hue"
        )
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=4,
        help=(
            "A inverse-stddev factor "
            "for the value and the saturation of the palette, "
            "which is of the weight of colors around the palette hue"
        )
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=2,
        help=(
            "A inverse-weight for the saturations "
            "of background, black and bright black colors"
        )
    )
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    main(args)
