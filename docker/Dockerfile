#FROM jaimeps/rl-gym:latest
# local copy 
FROM gcr.io/gdax-dnn/rl-base@sha256:9acd3a449e9460a150379a03c43e7a59d56cd0f82d7d4a2eeb8c4965fa31de56

ARG BUCKET_NAME
ENV BUCKET_NAME=${BUCKET_NAME}

ADD app /app

RUN chmod 777 /app/entrypoint.sh && \
pip install scikit-image && \
pip install keras==2.2.5 && \
pip install google-cloud-storage && \
pip install google-api-python-client && \
pip install oauth2client && \
apt-get update && \
apt-get install vim -y 

WORKDIR /app

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

