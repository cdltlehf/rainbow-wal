import argparse

import numpy as np
import pywal  # type: ignore

from . import (
    load_image, get_palettes, get_colors,
    DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_DELTA
)


def _test(
    filename: str,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float
) -> None:
    from matplotlib import pyplot as plt  # type: ignore

    image = load_image(filename)
    palette, debug_palette = get_palettes(image, alpha, beta, gamma, delta)
    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.imshow(image)

    plt.subplot(2, 1, 2)
    plt.axis("off")
    plt.imshow(np.concatenate((debug_palette, palette)))
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument(
        '--alpha',
        type=float,
        default=DEFAULT_ALPHA,
        help=(
            "A inverse-stddev factor for the hue of the palette, "
            "which is of the weight of colors around the primary hue."
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
        )
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=DEFAULT_GAMMA,
        help=(
            "A weight for the saturation of background, "
            "black and bright black colors."
        )
    )
    parser.add_argument(
        '--delta',
        type=float,
        default=DEFAULT_DELTA,
        help="A multiplicative factor for the lightness of the palette."
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

    if args.debug:
        _test(args.filename, args.alpha, args.beta, args.gamma, args.delta)
        exit(0)

    colors = get_colors(
        args.filename, args.alpha, args.beta, args.gamma, args.delta)
    pywal.wallpaper.change(colors["wallpaper"])
    pywal.sequences.send(colors, to_send=not args.s, vte_fix=args.vte)
    pywal.export.every(colors)


if __name__ == "__main__":
    main()
