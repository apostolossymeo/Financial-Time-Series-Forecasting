.PHONY: setup test figures train clean

setup:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	PYTHONPATH=. pytest -q

figures:
	PYTHONPATH=. python scripts/make_figures.py

train:
	PYTHONPATH=. python run.py

clean:
	rm -f results/*.keras results/*.csv
