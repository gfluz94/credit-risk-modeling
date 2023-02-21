FROM public.ecr.aws/lambda/python:3.9

COPY serve/app app

COPY ./models/artifacts models

RUN mv app/main.py .

COPY requirements.txt .

COPY ./credit-risk-modeling credit-risk-modeling

RUN cd credit-risk-modeling && \
	python setup.py install --prefix=.. && \
	cd .. && \
    rm requirements.txt && \
    rm -r credit-risk-modeling

CMD ["main.handler"]