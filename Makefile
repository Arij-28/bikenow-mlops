.PHONY: install quality prepare train pipeline mlflow predict api test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

quality:
	python src/data_quality.py

prepare:
	python src/prepare_data.py

train:
	python src/train_first_model.py

pipeline:
	python src/train_pipeline.py

mlflow:
	mlflow ui

predict:
	python src/predict_batch.py

api:
	uvicorn src.api:app --reload

test:
	pytest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
