# Arguments from FROM
ARG REGISTRY

FROM ${REGISTRY}/fxwatcher:ubuntu-python3

ARG handler_file=handler.py
ARG handler_name=Handler

ENV HANDLER_DIR=/openfx/handler
ENV HANDLER_FILE=${HANDLER_DIR}/${handler_file}
ENV HANDLER_NAME=${handler_name}
ENV PYTHONBUFFERED=0

RUN mkdir -p ${HANDLER_DIR}
WORKDIR ${HANDLER_DIR}

COPY ./requirements.txt ./
RUN pip install --upgrade setuptools
RUN pip install -r ./requirements.txt

COPY ./src/handler.py ./handler.py
COPY ./src/fairness ./fairness

RUN touch ${HANDLER_DIR}/__init__.py

RUN cp -r /openfx/executor/* ${HANDLER_DIR}

EXPOSE 50051
EXPOSE 50052

RUN export PYTHONIOENCODING=UTF-8
HEALTHCHECK --interval=5s CMD [ -e /tmp/.lock ] || exit 1

ENTRYPOINT ["python3"]
CMD ["server.py"]