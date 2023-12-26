.PHONY: default
default: run

.PHONY: run
run:
	rainbow_wal --filename "$$(get_wallpaper)"

.PHONY: debug
debug:
	rainbow_wal --filename "$$(get_wallpaper)" --debug
