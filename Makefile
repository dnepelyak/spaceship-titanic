.PHONY: install train test lint format clean

install:
	pip install -r requirements.txt

train:
	python src/train.py --n_trials 30 --n_folds 5

train-fast:
	python src/train.py --n_trials 10 --n_folds 3

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:
	black src/ tests/
	isort src/ tests/

mlflow-ui:
	mlflow ui --port 5000

app:
	streamlit run app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache mlruns/ models/submission.csv logs/
