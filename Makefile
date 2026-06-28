.PHONY: test test-core test-e2e coverage api-status api-serve api-smoke llm-status llm-driver-status llm-driver-smoke

PYTHON ?= python3
PYTHONPATH ?= .

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -B -m pytest -q -p no:cacheprovider

test-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -B -m pytest -q -p no:cacheprovider tests

test-e2e:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -B -m pytest -q -p no:cacheprovider tests

coverage:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -B -m pytest -q -p no:cacheprovider --cov=cognitive_evolve_runtime --cov-report=term-missing --cov-fail-under=80

api-status:
	$(PYTHON) scripts/cogev.py api status

api-serve:
	$(PYTHON) scripts/cogev.py api serve

api-smoke:
	$(PYTHON) scripts/cogev_api_smoke.py

llm-status:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/cogev.py llm status

llm-driver-status:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/cogev.py llm status

llm-driver-smoke:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/cogev.py llm smoke
