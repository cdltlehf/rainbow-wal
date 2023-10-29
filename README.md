# Rainbow-wal

Rainbow-wal is a color scheme generator from an image, which maintains the
ANSI-16 colors (black, red, green, yellow, blue, magenta, cyan and white.)


```
usage: main.py [-h] [--filename FILENAME] [--output OUTPUT] [--debug]

options:
  -h, --help           show this help message and exit
  --filename FILENAME  (default: ./resources/tuplips.png
  --output OUTPUT      (default: ~/.config/wal/colorschemes/dark/custom.json)
  --debug
```

The result JSON file can be used as a theme for
[pywal](https://github.com/dylanaraps/pywal/tree/master).
