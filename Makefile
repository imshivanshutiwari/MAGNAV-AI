PYTHON ?= python3

.PHONY: install run test lint dashboard clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) pipeline/main.py

dashboard:
	$(PYTHON) dashboard/app.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	$(PYTHON) -m flake8 . --max-line-length=120 --exclude=.git,__pycache__,notebooks

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .pytest_cache dist build *.egg-info
