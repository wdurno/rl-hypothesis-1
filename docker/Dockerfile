FROM jaimeps/rl-gym:latest

ARG BUCKET_NAME
ENV BUCKET_NAME=${BUCKET_NAME}

RUN pip install scikit-image && \
pip install google-cloud-storage && \
apt-get update && \
apt-get install vim -y 

ADD app /app 

WORKDIR /app

ENTRYPOINT bash /app/initial-rl.sh

