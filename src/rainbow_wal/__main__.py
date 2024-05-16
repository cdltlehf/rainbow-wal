import argparse
import subprocess

import numpy as np
import pywal  # type: ignore

from . import (
    load_image, get_palettes, get_colors,
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA,
    CHROMA_THRESHOLD,
    MINIMUM_CHROMA_CHROMATIC_COLOR,
    MINIMUM_LIGHTNESS_CHROMATIC_COLOR,
    MAXIMUM_LIGHTNESS_CHROMATIC_COLOR,
)


def get_wallpaper() -> str:
    script: str = (
        'tell app "finder" to get posix path of (get desktop picture as alias)'
    )
    wallpaper_path: str = subprocess.run(
        ['/usr/bin/osascript', '-e', script], capture_output=True
    ).stdout.decode('utf8').strip()
    return wallpaper_path


def _test(
    filename: str,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    chroma_threshold: float,
    minimum_chroma: float,
    minimum_lightness: float,
    maximum_lightness: float,
) -> None:
    from matplotlib import pyplot as plt  # type: ignore

    image = load_image(filename)
    palette, debug_palette = get_palettes(
        image,
        alpha, beta, gamma, delta,
        chroma_threshold, minimum_chroma,
        minimum_lightness, maximum_lightness,
    )
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.imshow(image)

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.imshow(np.concatenate((debug_palette, palette)))
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?')

    parser.add_argument(
        '--alpha',
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "A inverse-stddev factor for the hue of the palette, "
            "which is of the weight of colors around the primary hue. "
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=DEFAULT_BETA,
        help=(
            "A inverse-stddev factor "
            "for saturation and lightness of the palette, "
            "which is of the weight of colors around the palette hue."
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=DEFAULT_GAMMA,
        help=(
            "A weight for the saturation of background, "
            "black and bright black colors."
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        '--delta',
        type=float,
        default=DEFAULT_DELTA,
        help=(
            "A multiplicative factor for the lightness of the palette."
            "(default: %(default)s)"
        )
    )

    parser.add_argument(
        '--chroma-threshold',
        type=float,
        default=CHROMA_THRESHOLD,
        help="Chroma threshold for achromatic colors. (default: %(default)s)"
    )

    parser.add_argument(
        '--minimum-chroma',
        type=float,
        default=MINIMUM_CHROMA_CHROMATIC_COLOR,
        help="Minimum chroma for chromatic colors. (default: %(default)s)"
    )

    parser.add_argument(
        '--minimum-lightness',
        type=float,
        default=MINIMUM_LIGHTNESS_CHROMATIC_COLOR,
        help="Minimum lightness for chromatic colors. (default: %(default)s)"
    )

    parser.add_argument(
        '--maximum-lightness',
        type=float,
        default=MAXIMUM_LIGHTNESS_CHROMATIC_COLOR,
        help="Maximum lightness for chromatic colors. (default: %(default)s)"
    )

    parser.add_argument(
        '--unsafe',
        action='store_true',
        help="Use arguments chosen without consideration for readability."
    )

    parser.add_argument(
        '-s', action='store_true',
        help="Skip changing colors in terminals. (Pywal option)"
    )

    parser.add_argument(
        '--vte',
        action='store_true',
        help="Fix text-artifacts printed in VTE terminals. (Pywal option)"
    )

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.filename is None:
        args.filename = get_wallpaper()

    if args.unsafe:
        args.chroma_threshold = 0.0625
        args.minimum_chroma = 0.0625
        args.minimum_lightness = 0.0625
        args.maximum_lightness = 1 - 0.0625

    if args.debug:
        _test(
            args.filename,
            args.alpha, args.beta, args.gamma, args.delta,
            args.chroma_threshold, args.minimum_chroma,
            args.minimum_lightness, args.maximum_lightness,
        )
        exit(0)

    colors = get_colors(
        args.filename,
        args.alpha, args.beta, args.gamma, args.delta,
        args.chroma_threshold, args.minimum_chroma,
        args.minimum_lightness, args.maximum_lightness,
    )
    pywal.wallpaper.change(colors["wallpaper"])
    pywal.sequences.send(colors, to_send=not args.s, vte_fix=args.vte)
    pywal.export.every(colors)


if __name__ == "__main__":
    main()
