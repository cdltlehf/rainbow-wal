.PHONY: default
default: run

.PHONY: run
run:
	rainbow_wal --filename "$$(get_wallpaper)"

.PHONY: debug
debug:
	rainbow_wal --filename "$$(get_wallpaper)" --debug

.PHONY: install
install:
	pip install --user .

.PHONY: dev
dev:
	pip install --user --editable .
