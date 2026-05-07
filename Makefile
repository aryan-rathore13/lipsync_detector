PYTHON ?= ./.venv/bin/python

.PHONY: test eval report setup-third-party download-models

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

eval:
	$(PYTHON) scripts/evaluate.py data/manifests/sample_eval_manifest.csv

report:
	$(PYTHON) scripts/generate_eval_report.py $(RUN_DIR)

setup-third-party:
	bash scripts/setup_third_party.sh

download-models:
	bash scripts/download_models.sh
