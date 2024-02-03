.PHONY: default
default: run

.PHONY: run
run:
	rainbow-wal "$$(get_wallpaper)"

.PHONY: debug
debug:
	rainbow-wal "$$(get_wallpaper)" --debug

.PHONY: install
install:
	pip install --user .

.PHONY: dev
dev:
	pip install --user --editable .
