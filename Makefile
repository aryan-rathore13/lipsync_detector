PYTHON ?= ./.venv/bin/python

.PHONY: test setup-third-party download-models

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

setup-third-party:
	bash scripts/setup_third_party.sh

download-models:
	bash scripts/download_models.sh
