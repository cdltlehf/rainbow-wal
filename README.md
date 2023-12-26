# Rainbow-wal

Rainbow-wal is a color scheme generator from an image, which maintains the
ANSI-16 colors (black, red, green, yellow, blue, magenta, cyan and white.)


# Install

```bash
python3 -m pip install --user <this_project>
```

# Usage

```bash
usage: rainbow_wal [-h] [--filename FILENAME] [--alpha ALPHA] [--beta BETA]
                   [--gamma GAMMA] [-s] [--vte] [--debug]

options:
  -h, --help           show this help message and exit
  --filename FILENAME  (default: ./resources/tuplips.png)
  --alpha ALPHA        A inverse-stddev factor for the hue of the palette,
                       which is of the weight of colors around the primary
                       hue.
  --beta BETA          A inverse-stddev factor for saturation and lightness of
                       the palette, which is of the weight of colors around
                       the palette hue.
  --gamma GAMMA        A weight for the saturation of background, black and
                       bright black colors.
  -s                   Skip changing colors in terminals. (Pywal option)
  --vte                Fix text-artifacts printed in VTE terminals. (Pywal
                       option)
  --debug
```

The result JSON file can be used as a theme for
[pywal](https://github.com/dylanaraps/pywal/tree/master).
