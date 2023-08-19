# Rainbow-wal

Rainbow-wal is a color scheme generator from an image, which maintains the
ANSI-16 colors (black, red, green, yellow, blue, magenta, cyan and white.)


```
usage: main.py [-h] [--filename FILENAME] [--output OUTPUT] [--alpha ALPHA]
               [--beta BETA] [--gamma GAMMA]

options:
  -h, --help           show this help message and exit
  --filename FILENAME
  --output OUTPUT      (default: ~/.config/wal/colorschemes/dark/custom.json)
  --alpha ALPHA        A weight for how much to extract hues based on the
                       primary hue
  --beta BETA          A weight for how much to extract saturation and values
                       for the selected hue
  --gamma GAMMA        A weight for decreasing the hue of background, black
                       and bright black colors
```

The result JSON file can be used as a theme for
[pywal](https://github.com/dylanaraps/pywal/tree/master).
