FROM python:3.9-slim

ARG SUMMER_PROJECT_NAME="picam"
ARG WORKERS=1
ARG PHASE="dev"

ENV SUMMER_PROJECT_NAME ${SUMMER_PROJECT_NAME}
ENV SUMMER_DEPLOYMENT_PHASE ${PHASE}
ENV SUMMER_WORKERS ${WORKERS}

COPY ./requirements.txt /
RUN pip install -r /requirements.txt
RUN pip install gunicorn
RUN rm /requirements.txt

COPY ./${SUMMER_PROJECT_NAME} /app/${SUMMER_PROJECT_NAME}

WORKDIR /app

ENTRYPOINT gunicorn -b :5000 -w "${SUMMER_WORKERS}" --worker-class uvicorn.workers.UvicornWorker --access-logfile - --access-logformat '%(h)s %(l)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"' "${SUMMER_PROJECT_NAME}:create_app()" --timeout 20
