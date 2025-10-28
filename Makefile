# Berlin Road vs Crashes Project Makefile

.PHONY: help setup clean data pull features model viz

help:  ## Show this help message
	@echo "Berlin Road vs Crashes Project"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Set up the project environment
	conda env create -f environment.yml
	conda activate berlin-road-crash
	pip install python-dotenv

clean:  ## Clean temporary files and caches
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -delete
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/

data:  ## Pull and process data
	python src/fetch/mapillary_pull.py
	python src/fetch/berlin_crashes_pull.py
	python src/features/tiling.py

features:  ## Extract visual features from images
	python src/features/infer_yolo.py
	python src/features/aggregate.py

model:  ## Train and evaluate models
	python src/modeling/prepare_datasets.py
	python src/modeling/regress.py

viz:  ## Generate visualizations
	python src/viz/maps.py
	python src/viz/plots.py

all: data features model viz  ## Run the complete pipeline

# Development helpers
notebook:  ## Start Jupyter notebook server
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

test:  ## Run tests
	python -m pytest tests/ -v

lint:  ## Run code linting
	flake8 src/
	black --check src/
