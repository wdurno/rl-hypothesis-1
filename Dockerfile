FROM docker:dind

ENV GCLOUD_SDK_URL="https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz" \
    PATH="/opt/google-cloud-sdk/bin:/opt/spark/bin:${PATH}" \
    HADOOP_VERSION=2.7.3 \
    SPARK_VERSION=2.2.1 \
    PYSPARK_PYTHON=python3 \
    PYTHONPATH=/app/ai/app 

RUN apk --update --no-cache add \
        bash \
        ca-certificates \
        curl \
        openssl \
        python \
        python3 \
        gettext \
        openjdk8 && \
    wget -O - -q "${GCLOUD_SDK_URL}" | tar zxf - -C /opt && \
    ln -s /lib /lib64 && \
    gcloud config set core/disable_usage_reporting true && \
    gcloud config set component_manager/disable_update_check true && \
    gcloud config set metrics/environment github_docker_image && \
    gcloud --version && \
    rm -rf /tmp/* && rm -rf /opt/google-cloud-sdk/.install/.backup 

ADD app /app

WORKDIR /app

