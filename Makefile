.PHONY: test h1
test:
	python3 -m pytest
h1:
	PYTHONPATH=src python3 experiments/h1_certificates.py --samples 2000
