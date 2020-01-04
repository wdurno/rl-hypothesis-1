FROM jaimeps/rl-gym:latest

ARG BUCKET_NAME

RUN pip install scikit-image && \
pip install google-cloud-storage && \
apt-get install vim  

ADD app /app 

WORKDIR /app

