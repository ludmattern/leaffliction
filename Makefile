.PHONY: all install norm clean

all: install

install:
	pip install -r requirements.txt

norm:
	@python3 -m flake8 *.py && echo "✓ All Python files pass flake8 checks!" || echo "✗ Norm errors found"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
