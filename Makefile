all: test

test:
	source ./.venv/bin/activate; \
	python main.py --filename "$$(get_wallpaper)" && wal --theme custom; \
	qlmanage -p "$$(get_wallpaper)"
