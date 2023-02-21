FROM public.ecr.aws/lambda/python:3.9

COPY requirements.txt /ml/

COPY ./credit-risk-modeling /ml/credit-risk-modeling

RUN cd /ml/credit-risk-modeling && \
	python setup.py install --prefix=.. && \
	cd .. && \
    rm requirements.txt && \
    rm -r credit-risk-modeling

WORKDIR /ml/lib/python3.9/site-packages/

COPY serve/app app

COPY ./models/artifacts models

RUN mv app/main.py .

CMD ["main.handler"]