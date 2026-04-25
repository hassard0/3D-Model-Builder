.PHONY: help install preflight serve clean poses

help:
	@echo "3D Studio — common dev commands"
	@echo ""
	@echo "  make install     Run setup.sh (full install)"
	@echo "  make preflight   Diagnose env / weight / pose-gallery state"
	@echo "  make serve       Start the server (foreground)"
	@echo "  make poses       Regenerate static/poses/*.png"
	@echo "  make clean       Remove pose gallery and __pycache__"

install:
	@bash ./setup.sh

preflight:
	@python preflight.py

serve:
	@python app.py

poses:
	@python pose_gallery.py static/poses

clean:
	@rm -rf static/poses __pycache__
	@find . -name "*.pyc" -delete
