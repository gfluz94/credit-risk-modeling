install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	python3 -m black .

lint:
	python3 -m pylint --disable=R,C credit-risk-modeling/*.py

test:
	python3 -m pytest -vv --cov