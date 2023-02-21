FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt .

RUN python3.9 -m pip install -r requirements.txt -t .

COPY credit-risk-modeling/credit_risk_modeling credit_risk_modeling

COPY serve/app app

COPY ./models/artifacts models

RUN mv app/main.py .

CMD ["main.handler"]