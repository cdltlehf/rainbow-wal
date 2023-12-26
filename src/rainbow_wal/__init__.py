import os
import tempfile
from typing import (Optional, TypeVar, TypedDict, cast, )

import cv2  # type: ignore
import numpy as np
import pyheif  # type: ignore
from PIL import Image  # type: ignore
from hsluv import hsluv_to_rgb as _hsluv_to_rgb  # type: ignore
from hsluv import rgb_to_hsluv  # type: ignore
from numpy.typing import (NDArray, )
from scipy.stats import vonmises  # type: ignore

DEFAULT_ALPHA = 2.
DEFAULT_BETA = 8.
DEFAULT_GAMMA = 1.

CHROMA_THRESHOLD = 2 / 256

# https://www.w3.org/TR/WCAG20/#contrast-ratiodef
# we use the lightness instead of the luminance
CONTRAST_RATIO_CHROMATIC_COLOR = 7
CONTRAST_RATIO_BRIGHT_BLACK = 4.5

LIGHTNESS_BACKGROUND = 8 / 256
LIGHTNESS_BLACK = 1 / 256
LIGHTNESS_BRIGHT_BLACK = (
    CONTRAST_RATIO_BRIGHT_BLACK * (LIGHTNESS_BACKGROUND + 0.05) - 0.05)

MINIMUM_CHROMA_CHROMATIC_COLOR = 3 / 8
MINIMUM_LIGHTNESS_CHROMATIC_COLOR = (
    CONTRAST_RATIO_CHROMATIC_COLOR * (LIGHTNESS_BACKGROUND + 0.05) - 0.05)
MAXIMUM_LIGHTNESS_CHROMATIC_COLOR = 1 - MINIMUM_CHROMA_CHROMATIC_COLOR * 0.5
BRIGHT_RATIO = 1.1

MAXIMUM_IMAGE_SIZE = 1280 * 1024 / 32


class Colors(TypedDict):
    wallpaper: str
    alpha: str
    special: dict[str, str]
    colors: dict[str, str]


Float = np.float_
Radian = Float
TFloat = TypeVar('TFloat', NDArray[Float], float)


def hsl_to_rgb(_hsluv: NDArray[Float]) -> NDArray[np.uint8]:
    hue = np.degrees(_hsluv[0])
    saturation = min(max(_hsluv[1] * 100, 0), 100)
    lightness = min(max(_hsluv[2] * 100, 0), 100)
    hsluv = [hue, saturation, lightness]
    _rgb = [min(max(0, int(e * 256)), 255) for e in _hsluv_to_rgb(hsluv)]
    return np.array(_rgb, np.uint8)


def get_default_colors() -> NDArray[np.uint8]:
    # https://developer.apple.com/design/human-interface-guidelines/color#macOS-system-colors
    palette = np.zeros((2, 8, 3), np.uint8)

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


def get_circular_average(
    radians: NDArray[Float],
    weights: Optional[NDArray[Float]] = None
) -> Float:

    if weights is None:
        weights = np.full_like(radians, 1)

    sum_sin = np.sum(np.sin(radians) * weights)
    sum_cos = np.sum(np.cos(radians) * weights)

    circular_average = np.arctan2(sum_sin, sum_cos)
    return circular_average


def get_hue_weights(
    r_image_hues: NDArray[Radian],
    r_target_hue: Radian,
    *,
    factor: float = 1.0
) -> NDArray[Float]:
    stddev = np.pi / 6 / factor
    kappa = 1 / np.power(stddev, 2)
    _weights = [vonmises.pdf(e, kappa, r_target_hue) for e in r_image_hues]
    return np.array(_weights)


def get_hue_chromatic_color(
    r_image_hues: NDArray[Radian],
    r_target_hue: Radian,
    n_chromas: NDArray[Float],
    *,
    alpha: float
) -> Radian:
    hue_weights = get_hue_weights(r_image_hues, r_target_hue, factor=alpha)
    weights = hue_weights * n_chromas
    if sum(weights.flatten()) == 0:
        return r_target_hue
    hue_chromatic_color = get_circular_average(r_image_hues, weights)

    # -np.pi < hue_difference < np.pi
    hue_difference = hue_chromatic_color - r_target_hue
    if hue_difference > np.pi:
        hue_difference -= np.pi
    elif hue_difference < -np.pi:
        hue_difference += np.pi

    trim_boundary = np.pi / 12 - np.pi / 48
    if hue_difference > trim_boundary:
        hue_chromatic_color = r_target_hue + trim_boundary
    elif hue_difference < -trim_boundary:
        hue_chromatic_color = r_target_hue - trim_boundary

    return hue_chromatic_color


def get_achromatic_colors(
    r_image_hues: NDArray[Radian],
    r_primary_hue: Radian,
    n_saturations: NDArray[Float],
    n_lightnesses: list[float],
    *,
    gamma: float
) -> list[NDArray[Float]]:
    hue_weights = get_hue_weights(r_image_hues, r_primary_hue)
    n_saturation = np.average(n_saturations, weights=hue_weights) * gamma
    return [
        np.array([r_primary_hue, n_saturation, n_lightness])
        for n_lightness in n_lightnesses
    ]


def get_chromatic_color(
    image_hsl_tuple: tuple[NDArray[Radian], NDArray[Float], NDArray[Float]],
    n_chromas: NDArray[Float],
    r_hue_chromatic_color: Radian,
    *,
    beta: float
) -> Optional[NDArray[Float]]:

    r_image_hues, n_image_saturations, n_image_lightnesses = image_hsl_tuple
    hue_weights = get_hue_weights(
        r_image_hues, r_hue_chromatic_color, factor=beta)
    weights = hue_weights * n_chromas ** 2

    if sum(weights.flatten()) == 0:
        return None

    n_saturation = np.average(n_image_saturations, weights=weights)
    n_lightness = np.average(n_image_lightnesses, weights=weights)
    color = np.array([r_hue_chromatic_color, n_saturation, n_lightness])
    return color


def rgb_to_hex(color_rgb: NDArray[np.uint8]) -> str:
    r, g, b = tuple(color_rgb.tolist())
    color_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)
    return color_hex


def get_lighteness_factor(n_lightnesses: TFloat) -> TFloat:
    return 1 - np.absolute(2 * n_lightnesses - 1)


def get_chromas(n_saturations: TFloat, n_lightnesses: TFloat) -> TFloat:
    lightness_factor = get_lighteness_factor(n_lightnesses)
    n_chromas = n_saturations * lightness_factor
    return n_chromas


def get_chroma(color: NDArray[Float]) -> Float:
    lightness_factor = get_lighteness_factor(color[2])
    n_chromas = color[1] * lightness_factor
    return n_chromas


def print_color(color: NDArray[Float]):
    n_chroma = get_chroma(color)
    print(
        f'h: {np.degrees(color[0])}, '
        f's: {color[1]}, '
        f'l: {color[2]}, '
        f'chroma: {n_chroma}'
    )


def standardize_with_lightness(
    color: NDArray[Float],
    n_lightness: float,
) -> NDArray[Float]:
    n_chroma = get_chroma(color)
    lightness_factor = get_lighteness_factor(n_lightness)
    n_saturation = n_chroma / lightness_factor
    standardized_color = np.array([color[0], n_saturation, n_lightness])
    return standardized_color


def standardize_chromatic_color(
    color: NDArray[Float],
    bright: bool = False
) -> NDArray[Float]:
    standardized_color = color.copy()

    n_lightness = color[2]
    n_target_lightness = max(n_lightness, MINIMUM_LIGHTNESS_CHROMATIC_COLOR)
    n_target_lightness = min(
        n_target_lightness, MAXIMUM_LIGHTNESS_CHROMATIC_COLOR)
    if bright:
        n_target_lightness = n_target_lightness * BRIGHT_RATIO

    standardized_color = standardize_with_lightness(color, n_target_lightness)
    return standardized_color


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


def load_image(filename: str) -> NDArray[Float]:
    filename = os.path.expanduser(filename)
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
    image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return image


def get_palettes(
    image_rgb: NDArray[Float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    _image_hsl = [
        [rgb_to_hsluv(rgb.astype(float) / 256) for rgb in rgbs]
        for rgbs in image_rgb
    ]
    # hue: 0~360, saturation: 0~100, lightness: 0~100
    image_hsl = np.array(_image_hsl, float)
    _image_hsl_tuple = cast(
        tuple[NDArray[Float], NDArray[Float], NDArray[Float]],
        tuple(np.transpose(image_hsl, axes=(2, 0, 1)))
    )
    image_hues, image_saturations, image_lightnesses = _image_hsl_tuple
    r_image_hues = np.radians(image_hues)
    n_image_saturations = image_saturations / 100
    n_image_lightnesses = image_lightnesses / 100
    n_image_chromas = get_chromas(n_image_saturations, n_image_lightnesses)
    image_hsl_tuple = (r_image_hues, n_image_saturations, n_image_lightnesses)

    # rgb
    palette = get_default_colors()
    debug_palette = np.zeros((5, 8, 3), np.uint8)

    r_primary_hue = get_circular_average(r_image_hues, n_image_chromas)
    debug_palette[0][0] = hsl_to_rgb(np.array([r_primary_hue, 1, 0.5]))

    lightnesses_achromatic_colors = [
        LIGHTNESS_BACKGROUND, LIGHTNESS_BLACK, LIGHTNESS_BRIGHT_BLACK
    ]
    achromatic_colors = get_achromatic_colors(
        r_image_hues, r_primary_hue, n_image_saturations,
        lightnesses_achromatic_colors,
        gamma=gamma
    )

    palette[0][0] = hsl_to_rgb(achromatic_colors[1])
    palette[1][0] = hsl_to_rgb(achromatic_colors[2])

    primary_color = get_chromatic_color(
        image_hsl_tuple, n_image_chromas, r_primary_hue, beta=beta)
    n_primary_chroma = None
    if primary_color is not None:
        n_primary_chroma = get_chroma(primary_color)
        if n_primary_chroma < CHROMA_THRESHOLD:
            primary_color = None
    if primary_color is not None:
        debug_palette[1][0] = hsl_to_rgb(primary_color)

    background_color = hsl_to_rgb(achromatic_colors[0])
    debug_palette[2][0] = background_color

    base_chromatic_colors: list[Optional[NDArray[Float]]] = []
    if primary_color is not None:
        # NOTE: red is at 12 deg
        r_target_hues = np.radians(
            np.array([0, 120, 60, 240, 300, 180]) + 12, dtype=Radian)
        for i, r_target_hue in enumerate(r_target_hues, 1):
            r_hue_chromatic_color = get_hue_chromatic_color(
                r_image_hues, r_target_hue, n_image_chromas, alpha=alpha)
            base_chromatic_color = get_chromatic_color(
                image_hsl_tuple,
                n_image_chromas,
                r_hue_chromatic_color,
                beta=beta
            )
            base_chromatic_colors.append(base_chromatic_color)
            debug_palette[0][i] = hsl_to_rgb(np.array([r_target_hue, 1, 0.5]))
            debug_palette[1][i] = hsl_to_rgb(
                np.array([r_hue_chromatic_color, 1, 0.5]))
            if base_chromatic_color is not None:
                debug_palette[2][i] = hsl_to_rgb(base_chromatic_color)

    for i, base_chromatic_color in enumerate(base_chromatic_colors, 1):
        assert primary_color is not None
        if base_chromatic_color is None:
            continue

        chromatic_color = base_chromatic_color.copy()
        hue = chromatic_color[0]
        n_chroma_chromatic_color = get_chroma(chromatic_color)

        if n_chroma_chromatic_color < CHROMA_THRESHOLD:
            chromatic_color = np.array(
                [hue, primary_color[1], primary_color[2]])
        debug_palette[3][i] = hsl_to_rgb(chromatic_color)

        if n_chroma_chromatic_color < MINIMUM_CHROMA_CHROMATIC_COLOR:
            lightness = chromatic_color[2]
            lightness_factor = get_lighteness_factor(lightness)
            saturation = MINIMUM_CHROMA_CHROMATIC_COLOR / lightness_factor
            chromatic_color = np.array([hue, saturation, lightness])
        debug_palette[4][i] = hsl_to_rgb(chromatic_color)

        chromatic_color = standardize_chromatic_color(chromatic_color)
        bright_color = standardize_chromatic_color(chromatic_color, True)

        palette[0][i] = hsl_to_rgb(chromatic_color)
        palette[1][i] = hsl_to_rgb(bright_color)

    return palette, debug_palette


def get_colors(
    filename: str,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> Colors:

    image = load_image(filename)
    wallpaper = os.path.basename(filename)
    palette, debug_palette = get_palettes(image, alpha, beta, gamma)
    background_color = debug_palette[2][0]

    # hex
    colors = Colors(wallpaper=wallpaper, alpha='100', special={}, colors={})
    colors['special']['background'] = rgb_to_hex(background_color)
    colors['special']['foreground'] = rgb_to_hex(palette[0][7])
    colors['special']['cursor'] = rgb_to_hex(palette[0][7])

    for i in range(2):
        for j in range(8):
            colors['colors'][f'color{8 * i + j}'] = rgb_to_hex(palette[i][j])
    return colors
