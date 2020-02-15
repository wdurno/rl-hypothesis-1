FROM jaimeps/rl-gym:latest

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

