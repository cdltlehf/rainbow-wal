# Rainbow-wal

Rainbow-wal is a color scheme generator from an image, which maintains the
ANSI-16 colors (black, red, green, yellow, blue, magenta, cyan and white.)


```
usage: main.py [-h] [--filename FILENAME] [--output OUTPUT] [--alpha ALPHA]
               [--beta BETA] [--gamma GAMMA] [--debug]

options:
  -h, --help           show this help message and exit
  --filename FILENAME  (default: ./resources/tuplips.png
  --output OUTPUT      (default: ~/.config/wal/colorschemes/dark/custom.json)
  --alpha ALPHA        A stddev factor for the hue of palette color, which is
                       of the weight of colors around the primary hue
  --beta BETA          A stddev factor for the value and saturation of palette
                       color, which is of the weight of colors around the
                       palette hue
  --gamma GAMMA        A weight for the saturations of background, black and
                       bright black colors
  --debug
```

The result JSON file can be used as a theme for
[pywal](https://github.com/dylanaraps/pywal/tree/master).
