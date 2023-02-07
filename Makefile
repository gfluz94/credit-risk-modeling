install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt &&\
	cd credit-risk-modeling &&\
	python setup.py install &&\
	cd ..

format:
	python3 -m black .

lint:
	python3 -m pylint --disable=R,C credit-risk-modeling/credit_risk_modeling

test:
	python3 -m pytest -vv --cov