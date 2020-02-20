#FROM jaimeps/rl-gym:latest
# local copy 
FROM gcr.io/gdax-dnn/rl-base@sha256:9acd3a449e9460a150379a03c43e7a59d56cd0f82d7d4a2eeb8c4965fa31de56

ARG BUCKET_NAME
ENV BUCKET_NAME=${BUCKET_NAME}
ENV HADOOP_VERSION 2.7.3
ENV SPARK_VERSION 2.2.1
ENV PYSPARK_PYTHON python3

ADD app /app

# ml dependencies 
RUN chmod 777 /app/entrypoint.sh && \
pip install scikit-image && \
pip install keras==2.2.5 && \
pip install google-cloud-storage && \
pip install google-api-python-client && \
pip install oauth2client && \
apt-get update && \
apt-get install vim -y && \
apt-get install openjdk-8-jdk -y && \
# hadoop 
mkdir -p /opt && \
cd /opt && \
curl http://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz | \
    tar -zx hadoop-${HADOOP_VERSION}/lib/native && \
ln -s hadoop-${HADOOP_VERSION} hadoop && \
echo Hadoop ${HADOOP_VERSION} native libraries installed in /opt/hadoop/lib/native && \
# spark 
mkdir -p /opt && \
cd /opt && \
curl http://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz | \
    tar -zx && \
ln -s spark-${SPARK_VERSION}-bin-hadoop2.7 spark && \
echo Spark ${SPARK_VERSION} installed in /opt

ADD /app/spark/spark-defaults.conf /opt/spark/conf/spark-defaults.conf
ENV PATH $PATH:/opt/spark/bin 

WORKDIR /app

ENTRYPOINT ["bash", "/app/entrypoint.sh"]

